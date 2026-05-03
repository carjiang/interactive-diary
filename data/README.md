# Generate Synthetic Diary Dataset
Modify the location and number of entries generated in `data/gen_diary.py`
```bash
docker run --env-file .env -v $(pwd):/app -w /app your_username/your_docker_image_respository  \
python -u data/gen_diary.py
```

# Load Data Set
```bash
docker run mikono/id \
python -c "from data.schema_spec import load_dataset; load_dataset('data/synth_diary.jsonl', strict=False)"
```