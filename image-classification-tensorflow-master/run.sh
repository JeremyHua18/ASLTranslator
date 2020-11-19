python train.py \
  --bottleneck_dir=logs/pipelineB/bottlenecks \
  --how_many_training_steps=2000 \
  --model_dir=inception \
  --summaries_dir=logs/pipelineB/training_summaries/basic \
  --output_graph=logs/pipelineB/trained_graph.pb \
  --output_labels=logs/pipelineB/trained_labels.txt \
  --image_dir=./FINAL_DATASETS/pipelineB
