# Рекомендательная система
Задание - исследовать и разработать рекомендательную систему, ориентированную на пользовательскую активность в играх.
Был предоставлен датасет ml_test_rec_sys.parquet, который содержит ценную информацию о пользовательской активности.

## Цели задания

Изучение данных: Провести анализ предоставленных данных, выдвинуть гипотезы о пользовательском поведении и визуализируй свои находки.
Важность данных: Определить, какие аспекты данных наиболее значимы для рекомендаций и обосновать свой выбор.
Разработка системы: Создать рекомендательную систему, используя методы машинного обучения. Протестировать разные модели, оценить их качество и выбрать наилучший вариант.
Персонализация: Разработать механизм, предлагающий пользователю игры на основе его интересов и поведения.

--------
### 1. Изучение данных
Первым этапом разработки было изучение данных, получение среднего значения, медипнного и тд. Необходымых для первичного анализа данных.
Первым образом были найдены все возможные значения каждого параметра, а далее построены [графики](https://github.com/mrTechnik/MBS/tree/master/graphics/pre-proc) для каждого параметра.
```
                 Parameter          Mean   Median           STD         Variance
0                   UserID          50.0     50.0           0.0              0.0
1                GameTitle        1000.0   1000.0           0.0              0.0
2                   Rating       10000.0  10002.0     32.496154            844.8
3                      Age    819.672131    800.0    179.183028     31580.220371
4                   Gender  16666.666667  16000.0   1376.892637   1263888.888889
5                 Location   8333.333333   8275.0    468.686107    183055.555556
6                   Device  16666.666667  16800.0    275.378527     50555.555556
7            PlayTimeOfDay       12500.0  10400.0   4643.633635       16172500.0
8     TotalPlaytimeInHours    187.265918    100.0    236.529609      55736.71955
9          PurchaseHistory       25000.0  25000.0  13576.450199       92160000.0
10        InvolvementLevel       12500.0  12500.0    658.280589         325000.0
11              UserReview  16666.666667  16450.0    846.069343    477222.222222
12               GameGenre   7142.857143   7000.0   2911.389784   7265306.122449
13     GameUpdateFrequency  16666.666667  18000.0   2309.401077   3555555.555556
14          SocialActivity  16666.666667  16000.0   5033.222957  16888888.888889
15    LoadingTimeInSeconds   1219.512195   1000.0    524.985482    268887.566924
16  GameSettingsPreference  16666.666667  16000.0   5033.222957  16888888.888889

```
На основе этих значений можем сделать вывод, что UserID и GameTitlr это исключительно названия
Параметры: Rating, Device, Gender, InvolvementLevel, Location, UserReview - Имеют довольной слабый процент вбросов, медианы и средние значения не сильно отличаются, дисперсии слабые.
Параметры: Age, GameGenre, GameSettingsPreference, GameUpdateFrequency, LoadingTimeInSeconds, PlayTimeOfDay, PurchaseHistory, SocialActivity, TotalPlaytimeInHours - Имеют заметный процент вбросов, медианы и средние значения сильно отличаются, дисперсии не слабые.
Были выдвинуты предположения о корреляции GameTitle и Age, Rating, Location, PlayTimeOfDay, SocialActivity, LoadingTimeInSeconds

------
### 2. Важность данных
Для подтверждения/опровержения теорий и итогового нахождения коррелирующих параметров были использованы три метода:
  1. Метод коэффициентов (Пирсона, Спирмана, Канделла):
  ```
  Correlating params by (piercing, spearman, kendall) coefs:
0                  Rating
1               GameGenre
2     GameUpdateFrequency
3          SocialActivity
4    LoadingTimeInSeconds
Name: CorrFeature, dtype: object
  ```
  2. Метод OSL (метод наименьших квадратов):
```
  Correlating params by OLS:
0                 GameGenre
1       GameUpdateFrequency
2            SocialActivity
3      LoadingTimeInSeconds
4    GameSettingsPreference
Name: CorrFeature, dtype: object
```
  3. Метод Лассо:
  ```
  Correlating params by Lasso method:
0                     Rating
10                 GameGenre
11       GameUpdateFrequency
12            SocialActivity
13      LoadingTimeInSeconds
14    GameSettingsPreference
Name: CorrFeature, dtype: object
  ```
####По итогам этих трёх методов были выбраны 6 параметров: Rating, GameGenre, GameUpdateFrequency, SocialActivity, LoadingTimeInSeconds, GameSettingsPreference.
####В следствии чего делаем вывод, что наши гипотезы частично подтвердились. Далее перейдём к шагу по разработке рекомендательной системы и обучению моделей. 

------
### 3. Разработка рекомендательной системы
Для реализации этой задачи были выбраны 3 алгоритма:
  1. K-ближайших соседей
    Метод K-ближайших соседей — метрический алгоритм для автоматической классификации объектов или регрессии.
    В случае использования метода для классификации объект присваивается тому классу, который является наиболее распространённым среди K-соседей данного элемента, классы которых уже известны.
    В случае использования метода для регрессии, объекту присваивается среднее значение по K-ближайшим к нему объектам, значения которых уже известны.
    ```
    K-neighbours model accuracy is: 1.0
    ```
  3. Градиентый бустинг
    Градиентный бустинг — это техника машинного обучения для задач классификации и регрессии, которая строит модель предсказания в форме ансамбля слабых предсказывающих моделей, обычно деревьев решений.
    Цель любого алгоритма обучения с учителем — определить функцию потерь и минимизировать её.
    ```
    Gradient boost model accuracy is: 1.0
    ```
  4. Многослойный перцептрон
    Многослойный персептрон — это класс искусственных нейронных сетей прямого распространения, состоящих как минимум из трех слоёв: входного, скрытого и выходного. За исключением входных, все нейроны использует нелинейную функцию активации.
    При обучении MLP используется обучение с учителем и алгоритм обратного распространения ошибки.
     ```
    Epoch 10/10
    1/1 [==============================] - 0s 8ms/step - loss: 0.2287 - mae: 0.4511
    1/1 [==============================] - 0s 230ms/step - loss: 0.1854 - mae: 0.4044
    [0.18541474640369415, 0.4044051170349121]
     ```
  ####Как итог можно подвести, что с максимальной точностью работают методы K-ближайших соседей и Градиентный бучтинг, в то время как многослойный перцептрон совершает большой процент ошибок (По итогам испытаний большая погрешность обусловлена малым обучающим набором и несовершенством строения алгоритма, для решения данной задвчи требуется увеличение количества обучающих данных и доработка строения алгоритма).
  ####Однако по времени работы и обучения на первом месте стоит алгоритм k-ближайших соседей, далее многослойный перцептрон и градиентный бустинг.
  ####Как следствие, можно выделить алгоритм K-ближайших соседей, как лучший в данной подборке.
  
------
### 4. Персонализация
  Далее для доступа и возможности удалённого взаимодействия с данной рекомендательной системой было создано flask приложение с одним endpoint:
  ```
  Администратор@DESKTOP-1H0RJI7 MINGW64 ~
$ curl -X POST http://localhost:5000/recomendation -H "Content-Type: application/json" -d '{"UserID": "Usver_1", "Method": "mnn", "Data" : {"UserID":"user_991","GameTitle":"game_3","Rating":5,"Age":28,"Gender":"Other","Location":"Africa","Device":"PC","PlayTimeOfDay":"Morning","TotalPlaytimeInHours":25,"PurchaseHistory":"No","InvolvementLevel":"Expert","UserReview":"Positive","GameGenre":"RPG","GameUpdateFrequency":"Occasional","SocialActivity":"Low","LoadingTimeInSeconds":40,"GameSettingsPreference":"High Graphics"}}'
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   524  100    92  100   432    256   1203 --:--:-- --:--:-- --:--:--  1463{
  "Datetime": "2024-02-27 05:52:58.958007",
  "Result": "Game_21",
  "UseID": "Usver_1"
}


Администратор@DESKTOP-1H0RJI7 MINGW64 ~
$ curl -X POST http://localhost:5000/recomendation -H "Content-Type: application/json" -d '{"UserID": "Usver_1", "Method": "grad_boost_model", "Data" : {"UserID":"user_991","GameTitle":"game_3","Rating":5,"Age":28,"Gender":"Other","Location":"Africa","Device":"PC","PlayTimeOfDay":"Morning","TotalPlaytimeInHours":25,"PurchaseHistory":"No","InvolvementLevel":"Expert","UserReview":"Positive","GameGenre":"RPG","GameUpdateFrequency":"Occasional","SocialActivity":"Low","LoadingTimeInSeconds":40,"GameSettingsPreference":"High Graphics"}}'
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   536  100    91  100   445    248   1215 --:--:-- --:--:-- --:--:--  1468{
  "Datetime": "2024-02-27 05:53:05.044427",
  "Result": "Game_3",
  "UseID": "Usver_1"
}


Администратор@DESKTOP-1H0RJI7 MINGW64 ~
$ curl -X POST http://localhost:5000/recomendation -H "Content-Type: application/json" -d '{"UserID": "Usver_1", "Method": "k_neighbours_model", "Data" : {"UserID":"user_991","GameTitle":"game_3","Rating":5,"Age":28,"Gender":"Other","Location":"Africa","Device":"PC","PlayTimeOfDay":"Morning","TotalPlaytimeInHours":25,"PurchaseHistory":"No","InvolvementLevel":"Expert","UserReview":"Positive","GameGenre":"RPG","GameUpdateFrequency":"Occasional","SocialActivity":"Low","LoadingTimeInSeconds":40,"GameSettingsPreference":"High Graphics"}}'
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   539  100    92  100   447    333   1619 --:--:-- --:--:-- --:--:--  1952{
  "Datetime": "2024-02-27 05:53:09.353366",
  "Result": "Game_16",
  "UseID": "Usver_1"
}

  ```
####Данное приложение позволяет взаимодействовать клиенской части или, например, черех curl, postman и тд. для получения рекомендаций.

------
Дополнительно

Для тестирования, запуска и обучение всех аогоритмов рекомендуется вызывать:
```
$ python main.py
```
------
Для запуска, работы и тестирования backend приложения необходимо запустить:
```
$ python recomendation_system.py
```
