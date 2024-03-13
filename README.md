# ПРОЕКТ: Агентство недвижимости

К нам обратился представитель крупного агентства недвижимости со следующей проблемой:

*«Мои риелторы тратят катастрофически много времени на сортировку объявлений и поиск выгодных предложений. Поэтому их скорость реакции, да и, сказать по правде, качество анализа не дотягивают до уровня конкурентов. Это сказывается на наших финансовых показателях. Ваша задача — разработать модель, которая позволила бы обойти конкурентов по скорости и качеству совершения сделок. Вот датасет для работы».*

**Что необходимо сделать**: разработать сервис для предсказания стоимости домов на основе истории предложений.

## Данные
Данные содержат следующие признаки:

- `status` — статус продажи;
- `private pool`, `PrivatePool` — наличие бассейна;
- `ropertyType` — тип объекта;
- `street` — адрес объекта;
- `baths` — количество ванных;
- `homeFacts` — информация о строительстве объекта;
- `fireplace` — наличие камина;
- `city` — город объекта;
- `schools` — информация о школах рядом с объектом;
- `sqft` — площадь в квадратных футах;
- `zipcode` — почтовый индекс объекта;
- `beds` — количество спален;
- `state` — штат объекта;
- `stories` — количество этажей;
- `mls-id`, `MlsId` — идентификатор MLS;
- `target` — цена объекта (целевой признак).


## Структура 
Так как объем работы очень большой, то она делится на 2 этапа:

1. Обработка и подготовка данных в ноутбуке `preprocessing.ipynb`
2. Подбор модели, настройка параметров и подготовка к продакшену в `modeling.ipynb`
3. В папке `app` содержатся два файла `server.py` и `client.py`, которые реализуют деплой модели.


## Итог
Была создана модель типа `RandomForestRegressor`. Она предсказывает цену на недвидимость со средней абсолютной ошибкой в 7-8 тысяч единиц, а в процентном соотношении это примерно 0.15% от средней цены. Она сохранена в виде пайплайна в директории `app/models/realty_pipeline.pkl`.