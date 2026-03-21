"""
Custom middleware to allow API requests without redirecting to login.
"""
from django.utils.deprecation import MiddlewareMixin
from django.contrib.auth.models import AnonymousUser


class APIAuthBypassMiddleware(MiddlewareMixin):
    """
    Allow API proxy requests to pass through without login redirect.
    The FastAPI backend handles its own authentication if needed.
    The page itself is protected by @login_required.
    """
    
    def process_request(self, request):
        # If request is to /api/*, mark it to skip authentication redirects
        if request.path.startswith('/api/'):
            request._dont_enforce_csrf_checks = True
            # Store original user but don't require authentication for API proxy
            request._skip_login_required = True
        return None
    
    def process_view(self, request, view_func, view_args, view_kwargs):
        # Skip login_required decorator for API proxy requests
        if hasattr(request, '_skip_login_required') and request._skip_login_required:
            # Mark that this view should not redirect to login
            request.user._skip_auth_check = True
        return None
