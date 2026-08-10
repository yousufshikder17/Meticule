# Product control surface

The root URL serves a minimal same-origin control surface backed only by implemented APIs. It accepts a JWT issued outside the platform, stores it in browser session storage, calls `GET /me`, and uses the bearer token for subsequent requests. The server does not issue tokens, create initial tenants, or weaken membership validation. Bootstrap the first tenant administrator through an operator-controlled database/identity workflow.

The UI supports real agent create/update configuration, provider/tool inspection, run submission/list/status/cancellation/trace inspection, approval decisions, usage/cost visibility, deliberate memory writes/deletion, document ingestion/deletion/search, connector registration/discovery/revocation, tenant membership administration, and audit inspection. Unconfigured retrieval/connectors and unauthorized panels show the API error; they do not display synthetic success.

Role-sensitive APIs include:

- `tenant_admin`: membership listing, creation, and revocation; self-revocation is blocked.
- `usage_viewer` or `tenant_admin`: aggregate token/cost use.
- `audit_viewer` or `tenant_admin`: recent audit evidence.
- the approval's exact required role: pending approval inbox and existing decision enforcement.

The public shell contains no tenant data. Data endpoints remain authenticated, tenant-predicated, and inside the transaction-local RLS session. Membership administration has its own RLS policy in the initial schema baseline. Security headers use a same-origin CSP, deny framing, disable MIME sniffing, and omit referrers. The UI renders API values with `textContent`, not HTML.

Session storage is convenient for a local control surface but is not equivalent to hardened browser authentication: an XSS in same-origin code could read the token. Production deployments should prefer a reviewed external identity flow and short-lived tokens, and should consider a backend-for-frontend with secure cookies and CSRF protection. This stage does not implement signup, password reset, token issuance, refresh tokens, or account recovery.
