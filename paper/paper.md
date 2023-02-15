---
title: 'Torch PESQ - a PyTorch implementation of the Perceptual Evaluation of Speech Quality'
tags:
  - Python
  - audio
  - speech enhancement
authors:
  - name: Lorenz Schmidt
    equal-contrib: true
    affiliation: 1,2 # (Multiple affiliations must be quoted)
  - name: Nils Werner
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1,2
  - name: Nils Peters
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1,2
affiliations:
 - name: Friedrich-Alexander-Universität Erlangen-Nürnberg, Erlangen, Germany
   index: 1
 - name: International Audio Laboratories, Erlangen, Germany
   index: 2

date: 15 Feburary 2023

bibliography: paper.bib
---

# Summary

Auditory models are a family of methods used to mimic human listening and quickly evaluate audio according to human perception. Their parameters are derived from listening tests and help researchers in the audio domain to speed up their experiments. Furthermore they standardize comparison metrics for different audio estimates and make large-scale comparisons possible. On the downside, those metrics are often biased and targeted for a specific use-case. The alternative though are listening tests with human subjects. With enough samples they offer unbiased results but are time and resource consuming.

A major sub-field of hearing models is quality assessment of speech audio, especially of telephony systems. The human ear perceives audio non-linear in both loudness and frequency [@psychoacoustic]. Popular examples are the Perceptual Evaluation of Speech Quality (PESQ) [@pesq], Perceptual Objective Listening Quality Analysis (POLQA) [@polqa] or Virtual Speech Quality Objective Listener (ViSQOL) [@visqol] scores. They estimate a Mean Opinion Score (MOS) in range 1 (very bad) to 5 (very good) representing overall quality of the audio sample. Researchers often provide Matlab, Python or C reference implementation, which are not fit for machine learning tasks.

PyTorch [@pytorch] is a popular machine learning framework for the Python programming language based on the Torch library. It is common for vision tasks and found recent popularity for audio as well. One example where it is currently lacking are implementations of perceptual speech quality measures, which we target with this contribution. Our `torch-pesq` library adds a routine for PESQ MOS defined with PyTorch operators. Compared to the reference C implementation, `torch-pesq` allows batched PESQ MOS calculation on GPU accelerators. Furthermore it adds automatic differentiation support for the PESQ routine. As a result a machine learning model performance can be directly optimized with respect to the PESQ MOS.

# Statement of need

The field of speech enhancement relies on good metrics to compare models, and furthermore correct loss functions to achieve good metric performance. One approach models the MOS with an auxiliary neural network, for example a bidirectional long short-term memory model to achieve a high correlation with PESQ [@qualitynet]. MetricGan models [@metricgan; @qualitynet; @metricganplus] are using the PESQ score in their discriminator function, as it does not have to be differentiable. On the downside, this requires Generative Adversarial Networks (GAN) and does not work with other model structures. Finally, structural approaches try to copy the PESQ score operations [@pesq1; @pesq2] and replicate its output in that way. Unfortunately past proposals did not provide an implementation, preventing researcher to reproduce their experiments.

This open-source library will give researchers and developers a tested implementation of the PESQ score. It can be directly used as a loss function to improve model performance. Preliminary experiments have shown an improvement in validation results when compared to other loss functions. We combined the loss function of `torch-pesq` with Signal-Distortion Ratio [@sisdr]. Further information can be found on the repository readme page.

# Feature Overview and Comparison to Reference

With the above in mind, the library should offer a simple API to integrate into an existing PyTorch project. Two modes are present, the case for an estimate of the MOS and for the loss value - correlating negatively with MOS. Furthermore we made two modifications compared to the C reference implementation [@pesq1; @pesq2]. First, we did not implement time alignment between reference and degraded signal, as the input and output of a neural network are considered to be already aligned. Second, instead of doing energy alignment within the frequency domain, we use a short IIR filter to measure energy in band 300Hz to 3kHz [@pesq].

To make a proper comparison of the reference implementation to this library, we calculated correlation \autoref{fig:compare} for 30 different mixing factors of 10 speaker files [@vctk] and 9 noise scenarios [@demand].

![Correlation and maximum error of referenc to torch-pesq\label{fig:compare}](./compare_reference.png)

The results show a very high correlation (> 0.999) of the PESQ C reference implementation [@pesq] and `torch-pesq`. The maximum absolute error is below `0.18` MOS for most of the data points. The four point outliers are a result of the differences made for energy alignment and the following non-linear structure in disturbance estimation. 

# References
