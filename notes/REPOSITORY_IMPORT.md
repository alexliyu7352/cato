# R0 repository import provenance

The original three R0 commits were reconstructed through GitHub's Git Objects API after the ChatGPT GitHub App was granted access to this repository. GitHub regenerated commit metadata, so the recreated commit SHA-1 values differ from the local originals, while every corresponding Git tree SHA-1 matches exactly.

| Original local commit | Recreated GitHub commit | Exact Git tree SHA-1 |
| --- | --- | --- |
| `268bd362596fcc3991d4d7497dc94b5879385f63` | `df47b62f1590a8cf23eda532bd138c1a522c5ccf` | `5bfdf0b94d00bbd301f1efbe861ad296b63d14c3` |
| `df415adfb0ad4a9ac73210b335a5c29a16117c8c` | `4cc486bd9865a829f7fef426a0b9ef4807c81184` | `4aa25f303e00b0a91608d6bc3b26e73755bc8788` |
| `231ac5d6b7f89d5e80f1bf9d4d97cd66906b368c` | `1211644b09377f966827c1534efb1f013b5da78e` | `25729423a3614580200e8d7f0090ff33266cc341` |

The Git bundle at `history/R0-original-commits.bundle` contains the **actual original Git commit objects** and complete parent chain (SHA-1, author, timestamps). The archive was verified locally with `git bundle verify` before uploading.

To inspect or recover the original commit objects from a fresh clone:

```bash
git fetch ./history/R0-original-commits.bundle refs/heads/main:refs/remotes/original-r0/main
git log --oneline original-r0/main
```

The `main` branch follows the reconstructed history. The original commits are retained independently within the bundle. Do not force-push the original history over `main` without accounting for later contributions.

R0 uses synthetic engineering test data only; model checkpoints and large generated results are not committed. The Colab GPU and high-quality Ghibli-like teacher evaluations remain unverified.
