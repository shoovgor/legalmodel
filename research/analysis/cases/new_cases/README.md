# New Cases — Input Folder

Drop `.txt` files here (same format as the scraped live.shuukh.mn exports)
and run:

```bash
python3 predict_new_case.py
```

The script will:
1. Parse each `.txt` file using `shuukh_parser_v2`
2. Extract features (claim amount, party types, court type, narrative, judge)
3. Run predictions through the trained model
4. Print results to the terminal
5. Save a `predictions_output.csv` summary

**Note:** The model was trained on first-instance loan dispute cases
(`Бусад зээл`). Predictions for other dispute types may be unreliable.
