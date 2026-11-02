# Pre-upload checklist

1. Keep the independent educational-demo disclaimer at the top of README.md.
2. Keep all order records and policy examples synthetic. Do not add real customer data or copied confidential policies.
3. Never upload `.env`, tokens, passwords, private keys, logs, transcripts, model weights, or private datasets.
4. Review the complete Git history as well as the current working tree. Removing a secret from the latest file does not remove it from prior commits; rotate any exposed credential.
5. Review dependency names, versions, provenance, and licenses before publishing.
6. Keep the API bound to `127.0.0.1`. Do not expose `/add-document` or `/ws/chat` publicly without authentication, authorization, rate limiting, and a security review.
7. Do not use third-party logos or imply endorsement or official affiliation.
8. If GitHub has suspended the account, submit an honest appeal through GitHub’s official appeal process. Do not create another account to evade enforcement.
9. A successful local syntax/data check does not guarantee GitHub reinstatement or establish the reason for a suspension.
