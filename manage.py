#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    try:
        from django.core.management import execute_from_command_line

        # ✅ Commented out MongoDB connection check
        # if 'runserver' in sys.argv:
        #     from core.utils import verify_db_connection
        #     if not verify_db_connection():
        #         print("Warning: Application might not function correctly without database connection")
        #         user_input = input("Do you want to continue anyway? (y/n): ")
        #         if user_input.lower() != 'y':
        #             sys.exit(1)

        print("✅ Running with PostgreSQL backend only.")

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()



#!/usr/bin/env python
# """Django's command-line utility for administrative tasks."""
# import os
# import sys
# from dotenv import load_dotenv

# # Load .env variables
# load_dotenv()

# def main():
#     """Run administrative tasks."""
#     os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
#     try:
#         from django.core.management import execute_from_command_line
        
#         # Add this block to verify database connection
#         if 'runserver' in sys.argv:
#             from core.utils import verify_db_connection
#             if not verify_db_connection():
#                 print("Warning: Application might not function correctly without database connection")
#                 user_input = input("Do you want to continue anyway? (y/n): ")
#                 if user_input.lower() != 'y':
#                     sys.exit(1)
                    
#     except ImportError as exc:
#         raise ImportError(
#             "Couldn't import Django. Are you sure it's installed and "
#             "available on your PYTHONPATH environment variable? Did you "
#             "forget to activate a virtual environment?"
#         ) from exc
#     execute_from_command_line(sys.argv)

# if __name__ == '__main__':
#     main()