import argparse

from src.modeling import build_prediction_frame, load_bundle
from src.pipeline import run_pipeline


DEFAULT_DATA_PATH = "data/subsidies.xlsx"
DEFAULT_MODEL_PATH = "models/artifacts/subsidy_model.joblib"
DEFAULT_OUTPUT_PATH = "models/reports/predictions.csv"


def main():
    parser = argparse.ArgumentParser(description="Run batch prediction with trained model")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to xlsx data")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to saved model bundle",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save ranked predictions",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top rows to print in console",
    )

    args = parser.parse_args()

    bundle = load_bundle(args.model_path)
    df = run_pipeline(args.data_path)
    predictions = build_prediction_frame(
        df=df,
        tables=bundle["tables"],
        model=bundle["model"],
        blend_weights=bundle.get("blend_weights"),
        decision_threshold=bundle.get("decision_threshold", 0.5),
        probability_temperature=bundle.get("probability_temperature", 1.0),
    )
    from pathlib import Path

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Batch prediction complete")
    print("=" * 60)
    print(f"Model: {bundle['model_name']}")
    print(f"Rows scored: {len(predictions)}")
    print(f"Output saved to: {output_path}")
    print("")
    print(predictions.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
