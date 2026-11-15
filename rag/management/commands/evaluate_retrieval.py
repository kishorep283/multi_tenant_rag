from django.core.management.base import BaseCommand

from rag.evaluation.runner import run_evaluation, aggregate_metrics


class Command(BaseCommand):
    help = "Evaluate vector, keyword, hybrid and reranked retrieval."

    def handle(self, *args, **options):

        self.stdout.write("Starting retrieval evaluation...")

        results = run_evaluation()
        summary = aggregate_metrics(results)

        self.stdout.write("\nAggregate Retrieval Metrics\n")

        for stage, metrics in summary.items():
            self.stdout.write(f"\n{stage.upper()}")

            for metric, value in metrics.items():
                self.stdout.write(f"{metric}: {value:.4f}")

        self.stdout.write(
            self.style.SUCCESS("\nRetrieval evaluation completed.")
        )