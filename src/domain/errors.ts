export class NotFoundError extends Error { readonly status = 404; }
export class ConflictError extends Error { readonly status = 409; }
export class AuthorizationError extends Error { readonly status = 403; }
export class ConfigurationError extends Error { readonly status = 500; }
