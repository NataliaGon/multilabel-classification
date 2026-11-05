# EcoHubMap Multilabel Organization Classification

A research notebook for classifying organization descriptions into multiple sustainability-related categories. The project compares classical supervised text classifiers, optional multilingual encoder models and ensembles, and large language models using a shared evaluation protocol.

The main comparison is between a word TF-IDF + Linear SVM baseline and an improved word + character TF-IDF + Linear SVM model. An organization may receive more than one category.

## Dataset information

The study uses website-derived organization descriptions associated with [EcoHubMap](https://www.ecohubmap.com/). The notebook documents an article run with **7,347 usable records**, divided into **5,877 training records** and **1,470 held-out test records**, with **25 categories**.

The dataset is openly available on [Zenodo](https://doi.org/10.5281/zenodo.23949354) under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. It contains 7,347 organization records across 25 sustainability-related categories. Users may download and reuse the dataset with appropriate attribution.

Download `final_data.xlsx` and export it as a CSV before uploading it to the notebook. Preserve the original row order when reproducing the study’s train/test split and evaluation subset.

### Category catalog

The notebook defines the following 25 categories:

- agriculture
- air
- children
- consulting
- cosmetics
- eco-building
- ecological engineering
- ecotourism
- education
- environmental law
- equipment
- fashion
- finance
- food
- household products
- laboratory
- logistics
- packaging
- planting trees
- renewable energy
- renovation
- science
- transportation
- waste
- water

Labels are inferred from the uploaded CSV using `MultiLabelBinarizer`. For the documented study configuration, include all 25 categories and use the exact names above. The LLM section requires a description for every observed label; changing the catalogue requires corresponding code changes.

## Code information

[`Classification.ipynb`](Classification.ipynb) contains the complete workflow:

1. Environment setup and dependency-version recording.
2. CSV loading, cleaning, label encoding, and data-quality auditing.
3. A fixed multilabel-stratified train/test split.
4. Classical model comparison and threshold selection.
5. Held-out evaluation and overlap sensitivity analysis.
6. A separate deployment refit and model-package export.
7. Optional encoder, fine-tuning, and ensemble experiments.
8. Local Qwen zero-shot and few-shot classification.
9. Optional hosted OpenAI evaluation.
10. Aggregate metrics, research figures, and publication-results export.

Classical competitors include Logistic Regression, Complement Naive Bayes, a Logistic Regression classifier chain, and TF-IDF + truncated SVD + Extra Trees. Optional neural experiments include multilingual Sentence Transformer embeddings with Logistic Regression, label-description cosine similarity, frozen XLM-RoBERTa embeddings with an MLP, fine-tuned multilingual DistilBERT, and a classical/semantic ensemble.

The local generative model is `Qwen/Qwen2.5-3B-Instruct`. The hosted evaluation cell is configured for `gpt-5-mini` and `gpt-5.4`; execution depends on API access and model availability.

## Requirements

The notebook targets **Google Colab** and uses Colab-specific upload, download, and optional Google Drive functions. A fresh GPU runtime is recommended for the complete workflow. Classical TF-IDF models can run on a CPU; transformer and local LLM experiments require substantially more resources. The local Qwen configuration uses 4-bit loading suitable for a Colab T4.

The setup cell installs:

```text
iterative-stratification
transformers==4.57.6
sentence-transformers
accelerate
bitsandbytes
sentencepiece
requests
matplotlib
pandas
numpy
scikit-learn
joblib
```

PyTorch and IPython are also used and are normally available in Colab. Internet access is needed to install packages and download pretrained models. Hosted experiments additionally require an OpenAI API key, model access, and API billing.

Most package versions are not pinned. The notebook records installed versions in `environment_versions.txt`; retain this file with the results for reproducibility. Running outside Colab requires adapting `google.colab` calls and `/content` paths.

## Usage instructions

### 1. Open the notebook

[Open Classification.ipynb in Google Colab](https://colab.research.google.com/github/NataliaGon/multilabel-classification/blob/main/Classification.ipynb).

Select a fresh GPU runtime for the full experiment, then run the setup cell once before importing the remaining packages.

Alternatively, obtain a local copy:

```bash
git clone https://github.com/NataliaGon/multilabel-classification.git
cd multilabel-classification
```

The notebook is designed for Colab; cloning alone does not remove its Colab dependencies.

### 2. Prepare and load the dataset

Download `final_data.xlsx` from [Zenodo](https://doi.org/10.5281/zenodo.23949354) and export it as a UTF-8 CSV, preserving the column names, cell contents, and original row order.

Run the notebook’s dataset-upload cell and select the CSV. Alternatively, set `DATASET_PATH` in that cell to the location of your CSV before running it.

Review the data-quality audit and category counts before proceeding. The notebook excludes rows with empty descriptions or invalid or empty label lists.

### 3. Configure the experiments

Set each option in its defining configuration cell before executing dependent cells:

| Setting | Notebook default | Purpose |
| --- | --- | --- |
| `RUN_ADVANCED_EXPERIMENTS` | `True` | Enables extra classical competitors and optional encoder/ensemble experiments. |
| `RUN_LLM_EXPERIMENT` | `True` | Enables local Qwen inference. |
| `LLM_MODE` | `"full"` | Requests about 200 evaluation records; `"quick"` requests 10. |
| `RUN_OPENAI_EXPERIMENT` | `False` | Explicitly enables paid hosted-model requests. |
| `USE_GOOGLE_DRIVE_CHECKPOINT` | `False` | Saves LLM experiment artifacts to Google Drive when enabled. |
| `DOWNLOAD_FINAL_MODEL` | `True` | Downloads the deployment model package. |
| `AUTO_DOWNLOAD_LLM_ARTIFACTS` | `True` | Downloads the publication-results archive. |

For a lighter first run, set `RUN_ADVANCED_EXPERIMENTS = False` and `RUN_LLM_EXPERIMENT = False`, leaving hosted evaluation disabled. Continue through the configuration and reporting cells in order because later cells use variables they define.

To test local LLM inference on a smaller subset, set `LLM_MODE = "quick"`. Full-mode stratification and coverage repair can produce a different sample size from the requested count; the documented article run used **203 records**. Do not force exactly 200 when reproducing that run.

Hosted evaluation sends the selected descriptions to the API and incurs usage charges. Enable it only for data you are authorized to submit. The cell prompts for the API key without displaying it and saves separate resumable checkpoints for each model.

### 4. Run the analysis and save outputs

Run cells sequentially. Review cross-validation results, held-out metrics, overlap diagnostics, and common-subset comparisons.

Colab's `/content` directory is temporary. Enable Google Drive checkpoints before running the LLM configuration, or save private checkpoints before disconnecting. Preserve the same dataset and configuration when resuming an experiment.

## Methodology

### Data preparation

The notebook normalizes column names, removes technical `Unnamed:*` columns, strips surrounding text whitespace, parses and deduplicates labels within each record, and excludes empty texts and invalid or empty label lists. It creates a binary multilabel target matrix and audits duplicate texts, exact duplicate records, and conflicting labels.

Duplicate records are audited rather than automatically removed from the primary experiment. A separate sensitivity analysis excludes held-out records whose normalized text or website overlaps with training data.

### Model selection and evaluation

- Reserve approximately 20% of usable records using `MultilabelStratifiedShuffleSplit` with `random_state=42`.
- Use three-fold multilabel-stratified cross-validation on the training portion for classical candidate and threshold selection.
- Fit TF-IDF representations inside each training fold to avoid fitting vocabulary on validation records.
- Compare word unigrams/bigrams against combined word and character features. Character features use `char_wb` n-grams of lengths 3–5; both SVM models use balanced One-vs-Rest Linear SVM classifiers with `C=1.0`.
- Select classical thresholds from out-of-fold scores, prioritizing mean fold Macro F1 and using Micro F1 to break ties.
- Evaluate the selected configurations on the held-out test set using Macro F1, Micro F1, Samples F1, and per-category precision, recall, F1, and support.

Optional methods have their own settings: the cosine-similarity threshold is selected on training scores, the encoder-based Logistic Regression and MLP use cross-validation threshold selection, and fine-tuned DistilBERT uses a fixed probability threshold of 0.50.

### LLM comparison

We evaluate Local Qwen in zero-shot and few-shot modes. Five deterministic examples are selected exclusively from training data for the few-shot condition. Full-mode subset selection attempts to preserve multilabel distribution and ensure coverage of categories present in the held-out set.

Baseline and improved SVM predictions are restricted to the same selected records for comparison with the LLMs. The parser checks predicted labels against the catalogue and reports both parse-valid and strict-format rates without manual response correction. Hosted OpenAI evaluation uses the same selected subset.

### Deployment refit

After held-out assessment, the cross-validation-selected classical model is refitted on all usable records for deployment. This package is separate from the evaluation models and must not be used to measure performance on records included in that refit.

## Outputs and reproducibility

| Output | Contents |
| --- | --- |
| `environment_versions.txt` | Installed versions of key dependencies. |
| `ecohubmap_fields_classifier_improved.pkl` | Deployment package containing the fitted classical pipeline, label encoder, threshold, and metadata. The filename does not guarantee that the improved SVM won model selection. |
| LLM experiment directory | Private checkpoints, subset information, metrics, and figures. Defaults to `/content/EcoHubMap_LLM_Experiment`, or a Google Drive directory when enabled. |
| `EcoHubMap_publication_results.zip` | Aggregate common-subset metrics, per-label metrics, figures, run manifest, dependency versions, and hosted aggregate metrics when enabled. |

The notebook displays full held-out classical results and advanced experiment results; the publication ZIP is not an export of every runtime file or table.

The publication archive excludes organization descriptions, raw responses, private checkpoints, and model weights. Review artifacts before sharing. The deployment package's learned vocabulary may contain source-derived terms, so do not automatically treat it as redistributable.

Exact numerical reproduction requires the same cleaned corpus, row order, split, subset identifiers, settings, and compatible dependencies. Random seeds and deterministic local generation do not guarantee identical results across hardware, library versions, or future hosted-model changes. This README describes the workflow; it does not report newly rerun benchmark results.

## Citation

When using this code, cite the repository and identify the commit or release used. A repository citation can be written as:

> Noa Shtang *EcoHubMap Multilabel Organization Classification* [Computer software]. GitHub. https://github.com/NataliaGon/multilabel-classification. Specify the commit identifier and access date used in your work.

Also cite the corresponding research article when its final bibliographic details are available. If using a separately obtained dataset, cite that dataset's own record and comply with its access and reuse terms. This repository does not provide a public download of the study corpus.

## License

No `LICENSE` file was present in the repository when this README was prepared. A public repository alone does not establish an open-source reuse license. Contact the maintainer to clarify permissions before redistribution or reuse beyond rights otherwise granted by law or platform terms.

Dataset rights are separate from code rights. Third-party pretrained models and libraries remain subject to their respective licenses and terms.

## Contributions and support

Please use [GitHub Issues](https://github.com/NataliaGon/multilabel-classification/issues) to report bugs, discuss reproducibility questions, or propose changes. Include the relevant notebook section, dependency versions, configuration, and, where possible, a minimal synthetic example.

Pull requests should explain the change and any effect on preprocessing, data splits, thresholds, or evaluation comparability. Do not include API keys, source organization texts without permission, private checkpoints, or model artifacts without appropriate rights. Clearly distinguish changes to the research protocol from reproduction of the documented experiment.
