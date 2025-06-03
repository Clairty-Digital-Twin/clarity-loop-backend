# Third-Party Licenses and Notices

This document contains licenses and notices for third-party software components used in the CLARITY Digital Twin Platform.

## 📋 **Summary**

| Component | License | Location | Notes |
|-----------|---------|----------|-------|
| Pretrained Actigraphy Transformer (PAT) | CC BY-4.0 | `models/pat/` | Model weights only, used unmodified |
| CLARITY Platform Code | Apache 2.0 | Root directory | Our original codebase |

---

## 🧠 **AI Foundation Models**

### Pretrained Actigraphy Transformer (PAT)

**License:** Creative Commons Attribution 4.0 International (CC BY-4.0)

**Copyright:** © 2024 Franklin Y. Ruan, Aiwei Zhang, Jenny Oh, SouYoung Jin, Nicholas C. Jacobson

**Location:** `models/pat/`

**Source:** https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer

**Citation:**
```
Ruan, Franklin Y., Zhang, Aiwei, Oh, Jenny, Jin, SouYoung, and Jacobson, Nicholas C. 
"AI Foundation Models for Wearable Movement Data in Mental Health Research." 
arXiv:2411.15240 (2024). https://doi.org/10.48550/arXiv.2411.15240
```

**Usage:** We use the pre-trained model weights (PAT-S, PAT-M, PAT-L) unmodified for sleep pattern analysis and circadian rhythm detection in our Apache 2.0 licensed platform.

**Files:**
- `models/pat/PAT-S_29k_weights.h5` (1.1MB)
- `models/pat/PAT-M_29k_weights.h5` (3.8MB)  
- `models/pat/PAT-L_29k_weights.h5` (7.6MB)
- `models/pat/LICENSE` (Full CC BY-4.0 text)
- `models/pat/README.md` (Attribution details)

---

## ⚖️ **License Compatibility**

The combination of Apache 2.0 (our code) and CC BY-4.0 (PAT models) is legally compatible:

- ✅ **Commercial Use**: Both licenses allow commercial use
- ✅ **Attribution**: We provide proper attribution for CC BY-4.0 components
- ✅ **Separation**: Licenses are kept separate, not merged
- ✅ **Industry Standard**: This combination is widely used in open source

---

## 📜 **Full License Texts**

### Apache 2.0 License
The full Apache 2.0 license text for our platform code can be found in `LICENSE` at the root directory.

### CC BY-4.0 License  
The full Creative Commons Attribution 4.0 International license text for PAT components can be found in `models/pat/LICENSE`.

---

## 🙏 **Acknowledgments**

We gratefully acknowledge the Jacobson Lab at Dartmouth College for making their foundational research available under CC BY-4.0, enabling innovation in digital health platforms.

**Note:** This notice complies with both Apache 2.0 Section 4(d) requirements and CC BY-4.0 attribution requirements. 