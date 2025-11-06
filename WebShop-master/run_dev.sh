#!/bin/bash
#export FLASK_ENV=development
#python -m web_agent_site.app --log --attrs
export FLASK_ENV=production
python -m web_agent_site.app --log --attrs
