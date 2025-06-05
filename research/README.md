# Research References

This directory contains research repositories and reference materials used in the development of the Clarity Loop Backend.

## Pretrained-Actigraphy-Transformer

**Source**: [njacobsonlab/Pretrained-Actigraphy-Transformer](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer)
**Paper**: "AI Foundation Models for Wearable Movement Data in Mental Health Research" (arxiv:2411.15240)
**Paper Location**: `docs/literature/AI Foundation Models for Wearable Movement Data in Mental.pdf`
**License**: MIT License
**Purpose**: Reference implementation for our ML pipeline documentation and PAT model integration

### What We Used

- **Model Architecture**: Transformer encoder-decoder design with patch embeddings
- **Hyperparameters**: Production configurations for small/medium/large/huge model variants
- **Feature Extraction**: Actigraphy metrics implementation (sleep efficiency, circadian rhythm, etc.)
- **Training Methodology**: Pre-training and fine-tuning strategies
- **Performance Benchmarks**: Accuracy and latency targets for production deployment

### Integration Status

**Documentation Complete**: All implementation details extracted and documented in `docs/development/ml-pipeline.md`
**API Specification**: ML endpoints documented in `docs/api/ml-endpoints.md`
**Production Ready**: Configurations ready for FastAPI + Google Cloud deployment

### Citation

If you use this research in publications, please cite:

```
@article{jacobson2024foundation,
  title={AI Foundation Models for Wearable Movement Data in Mental Health Research},
  author={Jacobson, Nicholas C. and others},
  journal={arXiv preprint arXiv:2411.15240},
  year={2024}
}
```

### License Compliance

The original research is under MIT License, which allows commercial use, modification, and distribution. Our implementation follows all license requirements while adapting the research for production use in the Clarity Loop Backend.

## Future Research

This directory may contain additional research references as we expand the health AI capabilities of the platform.
