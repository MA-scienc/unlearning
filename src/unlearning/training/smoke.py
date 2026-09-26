from __future__ import annotations

from pathlib import Path

from unlearning.utils.io import ensure_dir, write_json


def run_smoke(output_dir: str | Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"status": "skipped", "reason": f"PyTorch unavailable: {exc}"}

    torch.manual_seed(7)
    out = ensure_dir(output_dir)
    model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Flatten(), torch.nn.Linear(32, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    s50_like = 2
    s100_like = 4
    lineage = {"S0": None}
    for step in range(1, s100_like + 1):
        x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        y = torch.tensor([1, 4])
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == s50_like:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, out / "S50.pt")
            lineage["S50"] = "S0"
        if step == s100_like:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, out / "S100.pt")
            lineage["S100"] = "S50"
    report = {
        "status": "passed",
        "steps": s100_like,
        "s50_like_step": s50_like,
        "s100_like_step": s100_like,
        "lineage": lineage,
        "note": "Synthetic smoke test only; not a research result.",
    }
    write_json(out / "smoke_report.json", report, overwrite=True)
    return report
