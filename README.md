# XRETAIL PIPELINE



Ce dépôt présente un pipeline novateur d'intelligence artificielle (IA) en vision par ordinateur, développé pour répondre au défi complexe de la numérisation des prix en masse dans les rayons des points de vente. Contrairement aux méthodes classiques de reconnaissance optique de caractères (OCR) telles que Tesseract, EAST, ou CRAFT, notre approche propose une solution plus efficace et précise pour l'extraction automatique des informations de prix, contribuant ainsi à l'automatisation des processus de gestion des stocks et des prix dans les environnements commerciaux, en particulier pour les fournisseurs.

Le dépôt aggrége un ensemble d'approches utiles pour la computer vision dans le retail, tel que l'analyse de flux clientèle par tracking anonyme, le scan et monitoring de prix via un pipeline ocr robuste,  mais également l'implantation rayon.

Je travaille seul sur le sujet depuis quelques années, mais le code à été développé depuis moins de 12 mois avec Pytorch et Tensorflow2. Le code est amené à beaucoup bougé et il manque pour le moment les modéles, notamment pour retinanet et croppad.

Le pipeline est développé dans une optique de développement CI/CD et MLOps, donc il faut vous attendre à voir arriver, une solution de stockage et de mise à jour de modéles, je ne connais pas encore l'état de l'art pour Tensorflow et Pytorch, mais une fois développé, le code sera mise à jour. Une solution de mise à jour de dataset, EDA des données, et entrainement à la volée,  une solution d'évaluation avec diverses métriques (IOU, mAP, Recall, etc ...)   