# Fake_News_Detection
Цель проекта - создать детектор фейковых новостей (для заголовков на русском языке).
В файле code.py содержится код, файл train.tsv содержит размеченные данные, файл test.tsv - данные, которые неободимо было разметить.
В файле preductions.tsv - результат разметки.

В качестве модели был выбран пассивно-агрессивный классификатор, так как его использование давало больший скор, чем другие линейные методы.

Использованные библиотеки: pandas, sklearn.

Статус проекта: нужается в доработке.
Планируется удаление стоп-слов (попытки удаления стоп-слов уменьшили скор), лемматизация.
