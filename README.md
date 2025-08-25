# AdSwap.ai

**AdSwap.ai** is a private research project exploring **AI-powered advertisement replacement in live sports broadcasts**.  
The goal is to automatically detect pitch-side boards and replace them with sponsor- or region-specific ads in real time.

## Status
- 🔒 Private repository – internal use only  
- 🚧 Work in progress (prototype stage)  
- 🎯 Current focus: detection, masking, and replacement of side-board advertisements  

## Planned Features
- Frame capture from video streams (NDI, SRT, file input)  
- Object detection with custom-trained models (Mask R-CNN, Detectron2)  
- Polygon masking & perspective correction  
- Ad image replacement with perspective warp  
- Frame-to-frame tracking for smooth overlays  

## Roadmap
- [ ] Initial model training for ad board segmentation  
- [ ] Real-time inference pipeline (GPU optimized)  
- [ ] Overlay replacement demo  
- [ ] Web-based preview interface  
- [ ] Performance tuning (TensorRT, CUDA)

---

*Internal prototype – not for public release.*
