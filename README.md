# mlops_made
Создание виртуального окружения (команды запускаются из корневой директории, unix-like системах):
~~
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
~~~

Обучение модели:
В корне вызывается команда (config_path - путь до .yaml файла):
~~~
python train.py config_path
python train.py configs/log_reg_no_regularization.yaml
~~~

Предсказание модели:
В корне вызывается команда (model_path - путь до модели, data_path - путь до csv-файла(важно наличие столбцов как при обучении),
output_path - путь для файла с предсказаниями)
~~~
python predict.py model_path data_path output_path
python predict.py models/model.pkl data/test/data.csv prediction.csv
~~~