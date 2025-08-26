import os

APP_ENV = os.getenv("APP_ENV", "prod").lower()
IS_DEV = APP_ENV.startswith("dev")
IS_PROD = APP_ENV.startswith("prod")
