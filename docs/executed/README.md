# Executed reading copies

These versioned snapshots can be read on GitHub without installing Python. Use the source link inside each notebook to run or adapt it. Relative imports in the source assume its original directory.

The manifests record source hashes, measured times and software/platform versions. They describe a particular run; outputs can change with data, seeds, versions or hardware.

- [Core XAI Exercises](Practice/Core_XAI_Exercises.ipynb)
- [Core XAI Solutions](Practice/Instructor/Core_XAI_Solutions.ipynb)
- [Dimensions](Theory/Dimensions.ipynb)
- [Causal Recourse](Tools/CausalRecourse_examples/Causal_Recourse.ipynb)
- [Evaluation and Sensitivity](Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb)
- [Regression Benchmark](Tools/Evaluation_examples/Regression_Benchmark.ipynb)
- [Interpretable Models Comparison](Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb)
- [Shared Image Attribution](Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb)
- [Mixed Data Workflow](Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb)
- [First XAI Comparison](Tools/Starter_examples/First_XAI_Comparison.ipynb)
- [Shared Text Comparison](Tools/TextComparison_examples/Shared_Text_Comparison.ipynb)

Regenerate using `scripts/check_notebooks.py`, then `scripts/publish_executed.py --input validation/<run>`. Publication rejects changed sources, failed runs and unexecuted cells.
