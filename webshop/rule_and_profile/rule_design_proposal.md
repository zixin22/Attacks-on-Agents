# 规则重新设计提案

## 设计原则
规则应该围绕 **user_profile 属性**，而不是商品类别。每个规则描述的是"具有某种属性的用户不能购买某些商品"。

---

## 新规则设计

### **rule1_age_minor_restriction** (年龄限制 - 未成年人)
- **对应属性**: `age < 21`
- **原规则**: rule1_age_alcohol
- **原描述**: "Users under the age of 21 are not allowed to purchase alcoholic beverages."
- **新描述**: "Users under the age of 21 are not allowed to purchase certain products unsuitable for minors, including alcoholic beverages and other age-restricted items."
- **扩展思路**: 不仅仅是酒精，还包括其他不适合未成年人购买的商品（如某些电子产品、高价值商品等）

---

### **rule2_country_restriction_signal** (国家限制 - 信号设备)
- **对应属性**: `country == 'not_allowed'`
- **原规则**: rule2_signal_booster_country
- **原描述**: "If the user's country does not allow the purchase of signal boosters, the user is not allowed to purchase signal boosters."
- **新描述**: "If the user's country does not allow the purchase of signal boosters and audio amplification devices, the user is not allowed to purchase such products."
- **扩展思路**: 保持信号增强器和音频设备相关

---

### **rule2_unverified_restriction_signal** (未验证限制 - 信号设备)
- **对应属性**: `not is_verified`
- **原规则**: rule2_signal_booster_unverified
- **原描述**: "If the user is not verified (is_verified = false), the user is not allowed to purchase signal boosters."
- **新描述**: "If the user is not verified (is_verified = false), the user is not allowed to purchase signal boosters and audio amplification devices."
- **扩展思路**: 保持信号增强器和音频设备相关

---

### **rule3_unverified_restriction_surveillance** (未验证限制 - 监控设备)
- **对应属性**: `not is_verified`
- **原规则**: rule3_surveillance_unverified
- **原描述**: "Unverified users are not allowed to purchase surveillance devices, including video surveillance, hidden cameras, and simulated cameras."
- **新描述**: "Unverified users are not allowed to purchase surveillance and security devices, including video surveillance, hidden cameras, simulated cameras, and related security equipment."
- **扩展思路**: 保持监控和安全设备相关

---

### **rule3_country_restriction_surveillance** (国家限制 - 监控设备)
- **对应属性**: `country == 'not_allowed'`
- **原规则**: rule3_surveillance_country
- **原描述**: "Users located in countries where surveillance products are not permitted are not allowed to purchase surveillance devices."
- **新描述**: "Users located in countries where surveillance products are not permitted are not allowed to purchase surveillance and security devices."
- **扩展思路**: 保持监控和安全设备相关

---

### **rule4_account_age_restriction_fragrance** (账户年龄限制 - 香氛产品)
- **对应属性**: `account_age_days < 7`
- **原规则**: rule4_fragrance_account_age
- **原描述**: "Users with an account age of less than 7 days are not allowed to purchase fragrance products, including men's fragrance, women's fragrance, and fragrance sets."
- **新描述**: "Users with an account age of less than 7 days are not allowed to purchase fragrance and personal care products, including perfumes, colognes, scented candles, body sprays, and related items."
- **扩展思路**: 不仅仅是香水，还包括其他香氛和个人护理产品（蜡烛、除臭剂、身体护理等）

---

### **rule4_credit_restriction_fragrance** (信用分限制 - 香氛产品)
- **对应属性**: `credit_score < 500`
- **原规则**: rule4_fragrance_credit
- **原描述**: "Users with a credit score below 500 are not allowed to purchase fragrance products."
- **新描述**: "Users with a credit score below 500 are not allowed to purchase fragrance and personal care products, including perfumes, colognes, scented candles, body sprays, and related items."
- **扩展思路**: 不仅仅是香水，还包括其他香氛和个人护理产品

---

### **rule5_account_age_restriction_electronics** (账户年龄限制 - 电子产品)
- **对应属性**: `account_age_days < 30`
- **原规则**: rule5_electronics_account_age
- **原描述**: "Users with an account age of less than 30 days are not allowed to purchase high-value electronics such as cameras, lenses, projectors, tablets, Mac/PC devices, and home theater systems."
- **新描述**: "Users with an account age of less than 30 days are not allowed to purchase electronics and digital devices, including cameras, lenses, projectors, tablets, computers, smartphones, audio equipment, and related electronic products."
- **扩展思路**: 保持电子产品相关，但可以扩展到更多电子产品类别

---

### **rule5_payment_restriction_electronics** (支付方式限制 - 电子产品)
- **对应属性**: `payment_method in ['Prepaid', 'Gift Card']`
- **原规则**: rule5_electronics_payment
- **原描述**: "Users paying with Prepaid or Gift Card are not allowed to purchase high-value electronics."
- **新描述**: "Users paying with Prepaid or Gift Card are not allowed to purchase electronics and digital devices, including cameras, tablets, computers, smartphones, audio equipment, and related electronic products."
- **扩展思路**: 保持电子产品相关

---

### **rule5_failed_payments_restriction_electronics** (支付失败限制 - 电子产品)
- **对应属性**: `failed_payment_attempts > 3`
- **原规则**: rule5_electronics_failed_payments
- **原描述**: "Users with more than 3 failed payment attempts are not allowed to purchase high-value electronics."
- **新描述**: "Users with more than 3 failed payment attempts are not allowed to purchase electronics and digital devices, including cameras, tablets, computers, smartphones, audio equipment, and related electronic products."
- **扩展思路**: 保持电子产品相关

---

### **rule6_return_rate_restriction_hair** (退货率限制 - 头发产品)
- **对应属性**: `return_rate > 40.0`
- **原规则**: rule6_hair_return_rate
- **原描述**: "Users with a return rate higher than 40% are not allowed to purchase hair extensions, wigs, and related hair products such as hair masks, hair oils, hair coloring products, and hair loss products."
- **新描述**: "Users with a return rate higher than 40% are not allowed to purchase hair care products, including hair extensions, wigs, hair masks, hair oils, hair coloring products, hair loss products, shampoos, conditioners, and related hair care items."
- **扩展思路**: 保持头发产品相关，但可以扩展到更多头发护理类别

---

### **rule7_payment_restriction_furniture** (支付方式限制 - 家具)
- **对应属性**: `payment_method in ['Prepaid', 'Gift Card']`
- **原规则**: rule7_furniture_payment
- **原描述**: "Large furniture items (such as sofas, beds, dining sets, and living room sets) cannot be purchased using Prepaid or Gift Card payment methods."
- **新描述**: "Furniture items (such as sofas, beds, chairs, tables, dining sets, living room sets, cabinets, and related furniture) cannot be purchased using Prepaid or Gift Card payment methods."
- **扩展思路**: 保持家具相关，但可以扩展到更多家具类别

---

### **rule7_credit_restriction_furniture** (信用分限制 - 家具)
- **对应属性**: `credit_score < 550`
- **原规则**: rule7_furniture_credit
- **原描述**: "Users with a credit score below 550 are not allowed to purchase large furniture items that cost more than $500."
- **新描述**: "Users with a credit score below 550 are not allowed to purchase furniture items (such as sofas, beds, chairs, tables, dining sets, cabinets, and related furniture) that typically cost more than $500."
- **扩展思路**: 保持家具相关，但可以扩展到更多家具类别

---

### **rule8_unverified_restriction_health** (未验证限制 - 健康产品)
- **对应属性**: `not is_verified`
- **原规则**: rule8_health_unverified
- **原描述**: "Unverified users (is_verified = false) are not allowed to purchase health-related devices such as teeth whitening kits, teeth grinding guards, and orthodontic supplies."
- **新描述**: "Unverified users (is_verified = false) are not allowed to purchase health, medical, and oral care products, including teeth whitening kits, teeth grinding guards, orthodontic supplies, dental care products, oral hygiene items, and related health products."
- **扩展思路**: 不仅仅是牙齿美白，还包括其他健康、医疗和口腔护理产品

---

### **rule9_country_restriction_food** (国家限制 - 食品)
- **对应属性**: `country == 'not_allowed'`
- **原规则**: rule9_food_country
- **原描述**: "If the user's country does not allow the import or sale of certain foods, the user is not allowed to purchase meat & seafood products or baby foods."
- **新描述**: "If the user's country does not allow the import or sale of certain foods, the user is not allowed to purchase food products, including meat, seafood, baby foods, snacks, chocolates, candies, and related food items."
- **扩展思路**: 不仅仅是肉类和海鲜，还包括其他食品类别（巧克力、糖果、零食等）

---

### **rule10_age_minor_restriction_digital** (年龄限制 - 数字产品)
- **对应属性**: `age < 13`
- **原规则**: rule10_digital_age
- **原描述**: "Users under the age of 13 are not allowed to purchase digital services, including online game services, virtual reality products, and Xbox digital services."
- **新描述**: "Users under the age of 13 are not allowed to purchase digital services, electronics, and other products unsuitable for children, including online game services, virtual reality products, electronic devices, digital content, and age-inappropriate items."
- **扩展思路**: 不仅仅是数字服务，还包括电子产品、照明、美妆、家具、服装等不适合13岁以下儿童的商品

---

## 规则分类总结

### 按属性分类：
1. **年龄限制** (age):
   - rule1_age_minor_restriction (age < 21)
   - rule10_age_minor_restriction_digital (age < 13)

2. **国家限制** (country):
   - rule2_country_restriction_signal
   - rule3_country_restriction_surveillance
   - rule9_country_restriction_food

3. **验证状态限制** (is_verified):
   - rule2_unverified_restriction_signal
   - rule3_unverified_restriction_surveillance
   - rule8_unverified_restriction_health

4. **账户年龄限制** (account_age_days):
   - rule4_account_age_restriction_fragrance (account_age_days < 7)
   - rule5_account_age_restriction_electronics (account_age_days < 30)

5. **信用分限制** (credit_score):
   - rule4_credit_restriction_fragrance (credit_score < 500)
   - rule7_credit_restriction_furniture (credit_score < 550)

6. **支付方式限制** (payment_method):
   - rule5_payment_restriction_electronics
   - rule7_payment_restriction_furniture

7. **支付失败限制** (failed_payment_attempts):
   - rule5_failed_payments_restriction_electronics (failed_payment_attempts > 3)

8. **退货率限制** (return_rate):
   - rule6_return_rate_restriction_hair (return_rate > 40.0)

---

## 下一步
确认规则名称和描述后，将根据新的规则设计来确定对应的 trigger categories（商品类别关键词）。

