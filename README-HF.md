---
title: Mango Vision
emoji: 🥭
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# MangoVision — Diagnóstico térmico de calidad de mango

Demo de despliegue (solo inferencia): sube una imagen y el modelo ResNet50
clasifica calidad/madurez y devuelve un Índice de Intensidad Térmica Relativa (ITR).

Las funciones de entrenamiento, HDFS, Kafka y métricas en vivo están
deshabilitadas en esta demo (requieren la infraestructura local de Big Data).
