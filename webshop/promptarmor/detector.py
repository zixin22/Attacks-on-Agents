"""
PromptArmor Detector: Detect and remove prompt injection
Following PromptArmor principle: LLM detection -> Extract injection content -> Fuzzy match removal
"""

import re
import json
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import openai

from .config import PromptArmorConfig


@dataclass
class DetectionResult:
    """Detection result data class"""
    text: str                    # Original text
    is_injected: bool           # Whether contains injection
    injection_content: str      # Extracted injection content
    cleaned_text: str          # Cleaned text
    confidence: float           # Confidence (0-1)
    raw_llm_response: str      # LLM raw response


class PromptArmorDetector:
    """PromptArmor Detector Main Class"""
    
    def __init__(self, config: Optional[PromptArmorConfig] = None):
        """
        Initialize detector
        
        Args:
            config: Configuration object, if None use default config
        """
        self.config = config or PromptArmorConfig()
        self.use_new_api = False  # Initialize flag
        self._init_openai_client()
        
    def _init_openai_client(self):
        """Initialize OpenAI API (compatible with both old and new API)"""
        try:
            # Read API Key (same path as main.py)
            if self.config.OPENAI_API_KEY_FILE.exists():
                with open(self.config.OPENAI_API_KEY_FILE, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
            else:
                api_key = os.getenv("OPENAI_API_KEY", "")
            
            if not api_key:
                raise ValueError(f"OpenAI API key not found in {self.config.OPENAI_API_KEY_FILE} or environment")
            
            # Set API key and base URL (compatible with old API format used in main.py)
            openai.api_key = api_key
            openai.api_base = self.config.OPENAI_API_BASE
            
            # Try to use new API (openai >= 1.0.0)
            try:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.config.OPENAI_API_BASE
                )
                self.use_new_api = True
            except Exception as e:
                # Do not fall back to old API when openai>=1.0.0 is installed.
                # Fail fast so users can fix their environment.
                raise RuntimeError(
                    f"Failed to initialize OpenAI client with new API: {e}"
                )
            
            if self.config.VERBOSE:
                print(f"✓ OpenAI client initialized (model: {self.config.DETECTION_MODEL}, base: {self.config.OPENAI_API_BASE}, new API)")
            
        except Exception as e:
            print(f"✗ Warning: Failed to initialize OpenAI client: {e}")
            raise
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM (compatible with both old and new OpenAI API)
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            LLM response text
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if self.use_new_api:
                    # Use new API (openai >= 1.0.0)
                    if self.config.DETECTION_MODEL == 'gpt-3.5-turbo-instruct':
                        full_prompt = f"{system_prompt}\n\n{user_prompt}"
                        response = self.client.completions.create(
                            model='gpt-3.5-turbo-instruct',
                            prompt=full_prompt,
                            temperature=self.config.DETECTION_TEMPERATURE,
                            max_tokens=self.config.DETECTION_MAX_TOKENS,
                            top_p=1,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            stop=["\n\n"]
                        )
                        return response.choices[0].text.strip()
                    else:
                        completion = self.client.chat.completions.create(
                            model=self.config.DETECTION_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=self.config.DETECTION_TEMPERATURE,
                            max_tokens=self.config.DETECTION_MAX_TOKENS,
                            top_p=1,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )
                        return completion.choices[0].message.content.strip()
                else:
                    raise RuntimeError(
                        "OpenAI client not initialized with new API. "
                        "Please fix your environment to use openai>=1.0.0 with OpenAI()."
                    )
                    
            except Exception as e:
                # Handle rate limit errors (compatible with both old and new API)
                error_str = str(e).lower()
                if 'rate limit' in error_str or 'ratelimit' in error_str:
                    wait_time = 5 * (attempt + 1)
                    if self.config.VERBOSE:
                        print(f"[Warning] OpenAI API rate limit, waiting {wait_time} seconds before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    if self.config.VERBOSE:
                        print(f"[Error] LLM call failed: {e}, waiting 3 seconds before retry ({attempt+1}/{max_retries})...")
                    time.sleep(3)
        
        # If all retries failed
        raise Exception(f"Failed to call LLM after {max_retries} retries")
    
    def detect_with_llm(self, text: str) -> Tuple[bool, str, str]:
        """
        Step 1 and 2: Use LLM to detect and extract injection content
        
        Following PromptArmor principle:
        1. Build prompt to let LLM judge if injection exists
        2. If detected, extract injection content
        
        Args:
            text: Text to detect
            
        Returns:
            (is_injected, injection_content, raw_response)
        """
        try:
            # Build user prompt (following PromptArmor principle)
            # User input is the data sample itself
            user_prompt = self.config.DETECTION_USER_PROMPT_TEMPLATE.format(text=text)
            
            # Call LLM
            raw_response = self._call_llm(
                system_prompt=self.config.DETECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            # Parse response
            is_injected = False
            injection_content = ""
            
            # Check if response contains "Yes" or "是"
            if "yes" in raw_response.lower() or "是" in raw_response or "true" in raw_response.lower():
                is_injected = True
                
                # Extract injection content (look for content after "Injection:")
                injection_match = re.search(
                    r'Injection:\s*(.+?)(?:\n\n|\n$|$)', 
                    raw_response, 
                    re.DOTALL | re.IGNORECASE
                )
                if injection_match:
                    injection_content = injection_match.group(1).strip()
                else:
                    # If "Injection:" marker not found, try to extract from response
                    lines = raw_response.split('\n')
                    injection_lines = []
                    found_injection = False
                    for line in lines:
                        if 'injection' in line.lower() or found_injection:
                            found_injection = True
                            if line.strip() and not line.strip().lower() in ['yes', '是', 'true', 'no', '否', 'false']:
                                injection_lines.append(line.strip())
                    if injection_lines:
                        injection_content = ' '.join(injection_lines)
            
            return is_injected, injection_content, raw_response
            
        except Exception as e:
            if self.config.VERBOSE:
                print(f"Error in LLM detection: {e}")
            return False, "", f"Error: {str(e)}"
    
    def extract_keywords(self, injection_text: str) -> List[str]:
        """
        Extract keywords from injection text (for fuzzy matching)
        
        Args:
            injection_text: Injection text
            
        Returns:
            List of keywords
        """
        # Remove punctuation and special characters
        cleaned = re.sub(r'[^\w\s]', ' ', injection_text)
        # Split into words
        words = cleaned.split()
        # Filter short words and common stop words
        keywords = [
            w.lower() 
            for w in words 
            if len(w) >= self.config.MIN_KEYWORD_LENGTH 
            and w.lower() not in self.config.STOP_WORDS
        ]
        return keywords
    
    def fuzzy_match_remove(self, text: str, injection_content: str) -> str:
        """
        Step 3: Use fuzzy matching to remove injection content
        
        Following PromptArmor principle:
        - Extract keywords from LLM-extracted injection content
        - Build regex pattern allowing arbitrary characters between keywords
        - Use regex to remove matched content
        
        Args:
            text: Original text
            injection_content: Injection content
            
        Returns:
            Cleaned text
        """
        if not injection_content:
            return text
        
        # Extract keywords
        keywords = self.extract_keywords(injection_content)
        
        if not keywords:
            # If no keywords, try exact match
            cleaned = text.replace(injection_content, "")
            return cleaned.strip()
        
        # Build fuzzy matching regex pattern
        # Allow arbitrary characters between keywords (including spaces, punctuation, etc.)
        pattern_parts = []
        for keyword in keywords:
            # Escape special characters
            escaped = re.escape(keyword)
            pattern_parts.append(escaped)
        
        # Build regex: allow arbitrary characters between keywords
        # Use .*? non-greedy matching to allow any characters
        fuzzy_pattern = r'.*?'.join(pattern_parts)
        
        # Try to match and remove (non-greedy matching, case-insensitive)
        cleaned = re.sub(fuzzy_pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up extra spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Merge multiple spaces into one
        cleaned = re.sub(r'\s*\.\s*\.', '.', cleaned)  # Clean consecutive periods
        cleaned = re.sub(r'\s*,\s*,', ',', cleaned)  # Clean consecutive commas
        cleaned = cleaned.strip()
        
        return cleaned
    
    def detect(self, text: str) -> DetectionResult:
        """
        Complete detection process
        
        Following PromptArmor principle:
        1. Use LLM to detect if injection exists
        2. If detected, extract injection content
        3. Use fuzzy matching to remove injection content
        
        Args:
            text: Text to detect
            
        Returns:
            DetectionResult object
        """
        if not text or not text.strip():
            return DetectionResult(
                text=text,
                is_injected=False,
                injection_content="",
                cleaned_text=text,
                confidence=0.0,
                raw_llm_response=""
            )
        
        # Step 1 and 2: Use LLM to detect and extract injection content
        is_injected, injection_content, raw_response = self.detect_with_llm(text)
        
        # Step 3: If injection detected, use fuzzy matching to remove
        cleaned_text = text
        if is_injected and injection_content:
            cleaned_text = self.fuzzy_match_remove(text, injection_content)
        
        # Calculate confidence (based on LLM response clarity)
        confidence = 0.0
        if is_injected:
            if "yes" in raw_response.lower() or "是" in raw_response.lower():
                confidence = 0.9
                if "Injection:" in raw_response:
                    confidence = 0.95  # Clearly extracted injection content
            else:
                confidence = 0.7  # Detected but not clear enough
        
        return DetectionResult(
            text=text,
            is_injected=is_injected,
            injection_content=injection_content,
            cleaned_text=cleaned_text,
            confidence=confidence,
            raw_llm_response=raw_response
        )
    
    def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        """
        Batch detect texts
        
        Args:
            texts: List of texts to detect
            
        Returns:
            List of DetectionResult
        """
        results = []
        for i, text in enumerate(texts):
            if self.config.VERBOSE:
                print(f"Detecting text {i+1}/{len(texts)}...")
            result = self.detect(text)
            results.append(result)
            # Avoid API rate limiting
            if i < len(texts) - 1:
                time.sleep(0.5)
        return results
    
    def save_results(self, results: List[DetectionResult], output_file: str):
        """
        Save detection results to file
        
        Args:
            results: List of detection results
            output_file: Output file path
        """
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        output_data = {
            'total': len(results),
            'detected': sum(1 for r in results if r.is_injected),
            'results': [asdict(r) for r in results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        if self.config.VERBOSE:
            print(f"✓ Results saved to {output_file}")
