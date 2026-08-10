import os
from pathlib import Path

real_dst = "datasets/RealFake/real/images"
fake_dst = "datasets/RealFake/fake/images"

real_files = sorted(Path(real_dst).glob("*.jpg"))[:100]
fake_files = sorted(Path(fake_dst).glob("*.jpg"))[:100]

with open("config/datasets/RealFake/real.txt", "w") as f:
    for path in real_files:
        f.write(path.as_posix() + "\n")

with open("config/datasets/RealFake/fake.txt", "w") as f:
    for path in fake_files:
        f.write(path.as_posix() + "\n")

print(f"Done! {len(real_files)} real, {len(fake_files)} fake")