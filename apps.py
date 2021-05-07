from django.apps import AppConfig


class AuthWebAppConfig(AppConfig):
    name = 'authwebapp'

    def ready(self):
        from . import signals
