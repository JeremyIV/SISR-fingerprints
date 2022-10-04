# Model Attribution Classifier Configs

This directory contains configs for all the models referenced in Section 3.3 (Classification Networks) and 4.1 (Model Attribution).

- `all_models/` contains configs for the classifiers from Section 3.3 (Classification Networks), 
- `custom_models/` contains configs for the three *custom model attribution classifiers* described in section 4.1. These classifier configs are identical; the only difference is the random seed, which is determined at runtime. `PRNU_SISR_custom_models.yaml` is used in Table 4.
- `pretrained_models` contains configs for the *pretrained model attribution classifiers*. **Important: these models are fine-tuned starting from the weights of the custom model attribution classifiers, so the custom model attribution classifiers must be trained first.**