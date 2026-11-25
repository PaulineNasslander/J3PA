"""Collect results across experiments and publish them to TensorBoard.

A run evaluates every (task, protocol) pair at several training-budget
milestones, e.g. "semantic / linear_probe" after 1k/5k/20k/50k steps. Aggregate
metrics are written as scalars indexed by the training step, so TensorBoard
draws one curve per experiment, and pointing it at several run directories
overlays the same curve for each model.

Per-class breakdowns are instead drawn as bar charts -- one bar per class -- with
``add_figure``, which is the only way to get a true bar chart into TensorBoard.
One chart is filed per milestone, so the image panel's step slider walks through
the training budgets.
"""

import matplotlib

matplotlib.use("Agg")  # no display in the container or on the cluster
import matplotlib.pyplot as plt


class ResultsRecorder:
    """Accumulates ``train_steps -> experiment -> metric -> value``."""

    def __init__(self):
        self.results: dict[int, dict[str, dict[str, float]]] = {}
        self.per_class: dict[int, dict[str, dict[str, dict]]] = {}
        self._written: set[tuple[int, str]] = set()

    def add(self, experiment: str, metrics: dict[str, float], train_steps: int, writer=None) -> None:
        """Record the scalar metrics of one experiment at one training budget.

        Args:
            experiment: Name such as ``"semantic/linear_probe"``.
            metrics: Flat metric name -> value. Nested dicts are treated as
                per-class breakdowns and kept aside for their own charts.
            train_steps: Number of optimizer steps taken before this evaluation.
            writer: Optional open `SummaryWriter`. Given one, the results go out
                immediately rather than waiting for `publish`, so the milestones
                a crashed run did finish are still on disk.
        """
        scalars, breakdowns = {}, {}
        for name, value in metrics.items():
            if isinstance(value, dict):
                breakdowns[name] = value
            else:
                scalars[name] = float(value)
        self.results.setdefault(train_steps, {})[experiment] = scalars
        if breakdowns:
            self.per_class.setdefault(train_steps, {})[experiment] = breakdowns
        if writer is not None:
            self._write(writer, experiment, scalars, train_steps)

    def _write(self, writer, experiment: str, scalars: dict[str, float], train_steps: int) -> None:
        """Write one experiment's scalars and charts, remembering not to repeat them."""
        for name, value in scalars.items():
            writer.add_scalar(f"final/{experiment}/{name}", value, train_steps)

        breakdowns = self.per_class.get(train_steps, {}).get(experiment, {})
        for chart_name, labels, values in _per_class_charts(breakdowns):
            figure = _bar_chart(
                labels=labels,
                values=values,
                title=f"{experiment} {chart_name} per class after {train_steps} steps",
                ylabel=chart_name,
            )
            writer.add_figure(f"per_class/{experiment}/{chart_name}", figure, train_steps)
            plt.close(figure)

        writer.flush()
        self._written.add((train_steps, experiment))

    def publish(self, writer) -> None:
        """Write whatever `add` has not streamed out yet, plus markdown tables.

        Args:
            writer: An open `SummaryWriter`.
        """
        if not self.results:
            print("No results to publish.")
            return

        recorded_steps = sorted(self.results)
        # Filing each result under its training budget turns the scalar panels
        # into one curve per experiment over 1k/5k/20k/50k steps. Anything `add`
        # already streamed out is skipped so the curves keep one point per step.
        for train_steps in recorded_steps:
            for experiment, metrics in self.results[train_steps].items():
                if (train_steps, experiment) not in self._written:
                    self._write(writer, experiment, metrics, train_steps)

        for train_steps in recorded_steps:
            writer.add_text("summary/table", self.as_markdown(train_steps), train_steps)
        writer.flush()

    def as_markdown(self, train_steps: int) -> str:
        """Render every experiment at one training budget as a markdown table."""
        results = self.results[train_steps]
        metric_names = _ordered_metric_names(results)
        lines = [
            f"**After {train_steps} training steps**",
            "",
            "| experiment | " + " | ".join(metric_names) + " |",
            "| --- |" + " --- |" * len(metric_names),
        ]
        for experiment, metrics in results.items():
            cells = [
                f"{metrics[name]:.4f}" if name in metrics else "-" for name in metric_names
            ]
            lines.append(f"| {experiment} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Dump the same tables to stdout so they land in the job log too."""
        print("\n" + "=" * 100)
        print("FINAL RESULTS")
        for train_steps in sorted(self.results):
            print("=" * 100)
            print(f"after {train_steps} training steps")
            for experiment, metrics in self.results[train_steps].items():
                formatted = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"{experiment:<40} {formatted}")
        print("=" * 100 + "\n")


def _ordered_metric_names(results: dict[str, dict[str, float]]) -> list[str]:
    """Metric names in first-seen order, so the tables stay stable across runs."""
    names = []
    for metrics in results.values():
        for name in metrics:
            if name not in names:
                names.append(name)
    return names


def _per_class_charts(breakdowns: dict[str, dict]) -> list[tuple[str, list[str], list[float]]]:
    """Split per-class breakdowns into one ``(name, labels, values)`` chart each.

    Handles both ``{"iou": {class: value}}`` and the nested panoptic form
    ``{"per_class": {class: {"PQ": .., "SQ": ..}}}``.
    """
    charts = []
    for breakdown_name, class_values in breakdowns.items():
        if not class_values:
            continue
        first_value = next(iter(class_values.values()))
        if isinstance(first_value, dict):
            for sub_metric in sorted(first_value):
                charts.append(
                    (sub_metric, list(class_values), [v[sub_metric] for v in class_values.values()])
                )
        else:
            charts.append((breakdown_name, list(class_values), list(class_values.values())))
    return charts


def _bar_chart(labels: list[str], values: list[float], title: str, ylabel: str):
    """One bar per label, value printed above it."""
    figure, axes = plt.subplots(figsize=(max(4.0, 0.9 * len(labels) + 2.0), 4.5))
    positions = range(len(labels))
    bars = axes.bar(positions, values, color="#4C72B0")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, value in zip(bars, values, strict=True):
        axes.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    figure.tight_layout()
    return figure
