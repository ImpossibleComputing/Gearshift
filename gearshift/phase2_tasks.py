"""Visible task contracts are explicitly separate from evaluation-only material."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace


CONTRACTS = {
    'arithmetic': 'Give only the final numerical answer, without explanation.',
    'code': 'Implement the function requested in the original task. Return the complete Python function and any required imports, with no explanatory prose. Do not call the function.',
    'evidence': 'Write the requested evidence-grounded decision report. Preserve the relevant facts, calculations, qualifications, and source IDs. Explain the decision and risks. Do not invent facts.',
    'writing': 'Write the artifact requested in the original brief for its stated audience. Preserve its constraints and facts. Produce the finished artifact, without discussing your reasoning process.'}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    split: str
    cluster_id: str
    prompt: str
    contract: str
    reasoning_budget: int
    answer_budget: int
    grader: str
    hidden: dict
    metadata: dict

    def visible(self):
        return {k:v for k,v in asdict(self).items() if k not in ['hidden','grader','metadata']}

    def generation_text(self):
        return self.prompt + '\n\nOutput contract: ' + self.contract


def from_dict(obj):
    return TaskSpec(**obj)


def apply_budgets(cfg,tasks):
    caps=cfg.get('source_reasoning_caps',{})
    return [replace(t,reasoning_budget=caps.get(t.family,t.reasoning_budget)) for t in tasks]
