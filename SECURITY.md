# Security and privacy

- Store credentials only in environment variables.
- Never commit `.env`, `api.txt`, keys, authorization headers, raw billing
  responses, or complete provider responses from private experiments.
- Run the checks in `PUBLIC_RELEASE_CHECKLIST.md` before every public release.
- If a credential has ever been committed, revoke it; deleting the file is not
  sufficient.
- Report suspected credential exposure privately to the repository maintainers.
