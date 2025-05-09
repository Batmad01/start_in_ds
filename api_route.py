import asyncio
from typing import Union, Dict, List
from fastapi import APIRouter, HTTPException
from http import HTTPStatus
from pydantic import BaseModel, field_validator
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

# Роутер, в котором мы определяем все эндпоинты для работы с моделями (fit, load, predict, и т.д.)
router = APIRouter()

# Глобальные хранилища состояний:
# models - словарь, хранящий обученные модели по их id
# current_model_id - идентификатор модели, которая в данный момент загружена для предикта
models = {}
current_model_id = None


# Все необходимые классы
class ValidationError(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class HTTPValidationError(BaseModel):
    detail: List[ValidationError]


class ModelConfig(BaseModel):
    # Конфигурация для обучения модели:
    # id - уникальный идентификатор модели,
    # ml_model_type - тип ML модели: "linear" или "logistic"
    # hyperparameters - гиперпараметры для модели
    id: str
    ml_model_type: str
    hyperparameters: dict


class FitRequest(BaseModel):
    # Запрос на обучение модели. Содержит:
    # X - двумерный массив чисел (features),
    # y - массив целевых значений,
    # config - конфиг модели (ModelConfig).
    X: List[List[float]]
    y: List[float]
    config: ModelConfig

    @field_validator('X')
    def validate_X(cls, v):
        # Проверяем, что X - это список списков чисел.
        if not all(isinstance(row, list) for row in v):
            raise ValueError("X must be a list of lists")
        return v

    @field_validator('y')
    def validate_y(cls, v):
        # Проверяем, что y - список чисел (float или int).
        if not all(isinstance(item, (float, int)) for item in v):
            raise ValueError("y must be a list of floats/ints")
        return v


class FitResponse(BaseModel):
    # Ответ при успешном обучении модели
    message: str


class LoadRequest(BaseModel):
    # Запрос на загрузку модели для предикта. Содержит только идентификатор модели.
    id: str


class LoadResponse(BaseModel):
    # Ответ при успешной загрузке модели
    message: str


class UnloadResponse(BaseModel):
    # Ответ при выгрузке модели
    message: str


class RemoveResponse(BaseModel):
    # Ответ при удалении модели или всех моделей
    message: str


class ModelItem(BaseModel):
    # Элемент списка моделей - содержит id и type модели
    id: str
    type: str


class ModelListResponse(BaseModel):
    # Ответ при запросе списка моделей. Возвращает массив объектов ModelItem.
    models: List[ModelItem]


class PredictRequest(BaseModel):
    # Запрос на предикт. Содержит:
    # id - идентификатор модели, для которой нужен предикт (должна быть загружена),
    # X - данные для предсказания (список списков чисел).
    id: str
    X: List[List[float]]

    @field_validator('X')
    def validate_predict_X(cls, v):
        # Проверяем, что X корректен для предикта (список списков чисел)
        if not all(isinstance(row, list) for row in v):
            raise ValueError("X must be a list of lists")
        return v


class PredictionResponse(BaseModel):
    # Ответ на предикт. predictions - объект (dict), значения которого - массивы чисел.
    predictions: Dict[str, List[float]]


class StatusResponse(BaseModel):
    # Ответ для статуса сервиса
    status: str
# --------------------------------------------------------------------


@router.post("/fit", response_model=FitResponse, status_code=HTTPStatus.CREATED, summary="Fit",
             description="Обучает модель с использованием переданных параметров конфигурации.")
async def fit(request: FitRequest):
    """
    Обучение новой модели по данным X, y и конфигурации config.
    - Проверяем, что модель с таким id не существует.
    - Тип модели определяем по config.ml_model_type (linear или logistic).
    - Запускаем длительное обучение (имитация 60 секунд с asyncio.sleep).
    - Обучаем модель с помощью sklearn.
    - Сохраняем обученную модель в памяти (models).
    """

    model_id = request.config.id
    ml_model_type = request.config.ml_model_type
    hyperparams = request.config.hyperparameters

    # Проверяем отсутствие дубликатов модели
    if model_id in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' already exists")

    # Проверяем, что тип модели подходит ("linear" или "logistic")
    if ml_model_type not in ["linear", "logistic"]:
        raise HTTPException(status_code=400, detail="ml_model_type must be 'linear' or 'logistic'")

    X = np.array(request.X)
    y = np.array(request.y)

    # Инициализируем модель в зависимости от типа
    if ml_model_type == "linear":
        model = LinearRegression(**hyperparams)
    else:
        model = LogisticRegression(**hyperparams, max_iter=1000)

    # Имитация длительного обучения (например, тяжёлая тренировка)
    await asyncio.sleep(60)
    model.fit(X, y)

    # Сохраняем модель в словарь
    models[model_id] = {"type": ml_model_type, "model": model}

    return FitResponse(message=f"Model '{model_id}' trained and saved")


@router.post("/load", response_model=LoadResponse, summary="Load",
             description="Загружает модель для выполнения предсказаний.")
async def load(request: LoadRequest):
    """
    Загрузка модели в контекст для дальнейшего предикта.
    - Проверяем, что модель с указанным id существует.
    - Устанавливаем ее как текущую загруженную модель (current_model_id).
    """
    global current_model_id
    model_id = request.id
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    current_model_id = model_id
    return LoadResponse(message=f"Model '{model_id}' loaded")


@router.post("/predict", response_model=PredictionResponse, summary="Predict",
             description="Выполняет предсказания используя предоставленные данные.")
async def predict(request: PredictRequest):
    """
    Предикт на загруженной модели.
    - Проверяем, что есть загруженная модель (current_model_id).
    - Проверяем, что id в запросе совпадает с загруженной моделью.
    - Если всё хорошо, вызываем predict на модели.
    - Возвращаем предсказания в формате, указанном схемой.
    """
    global current_model_id

    if current_model_id is None:
        raise HTTPException(status_code=400, detail="No model is currently loaded.")

    if request.id != current_model_id:
        # Идентификатор в запросе не совпадает с загруженной моделью
        raise HTTPException(status_code=400,
                            detail=f"Модель'{request.id}' не совпадает с '{current_model_id}'")

    if request.id not in models:
        # Модель исчезла из списка, возможно была удалена
        raise HTTPException(status_code=404, detail=f"Model '{request.id}' not found")

    model_info = models[request.id]
    model = model_info["model"]
    X = np.array(request.X)

    try:
        preds = model.predict(X)
    except Exception as e:
        # В случае ошибок при предикте, возвращаем 400 с текстом ошибки
        raise HTTPException(status_code=400, detail=str(e))

    # Формируем predictions как dict с ключом "values"
    return PredictionResponse(predictions={"values": preds.tolist()})


@router.post("/unload", response_model=UnloadResponse, summary="Unload",
             description="Выгружает модель из памяти.")
async def unload():
    """
    Выгрузка текущей модели.
    - Если модель загружена, сбрасываем current_model_id.
    - Если нет загруженной модели, возвращаем сообщение, что загруженной модели не было.
    """
    global current_model_id
    if current_model_id is None:
        # Нет загруженной модели, возвращаем просто сообщение.
        return UnloadResponse(message="No model was loaded")
    unloaded_model = current_model_id
    current_model_id = None
    return UnloadResponse(message=f"Model '{unloaded_model}' unloaded")


@router.get("/list_models", response_model=ModelListResponse, summary="List Models",
            description="Возвращает список всех обученных моделей.")
async def list_models():
    """
    Получение списка всех обученных моделей.
    Возвращаем список [{id: ..., type: ...}, ...].
    """
    model_list = [
        ModelItem(id=m_id, type=m_info["type"])
        for m_id, m_info in models.items()
    ]
    return ModelListResponse(models=model_list)


@router.delete("/remove/{model_id}", response_model=RemoveResponse, summary="Remove",
               description="Удаляет указанную модель из хранилища.")
async def remove(model_id: str):
    """
    Удаление конкретной модели по идентификатору.
    - Если модель загружена (current_model_id), и она совпадает с удаляемой, то выгружаем её.
    - Если модель с таким id не найдена, возвращаем 404.
    - Иначе удаляем её из словаря.
    """
    global current_model_id
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    if current_model_id == model_id:
        current_model_id = None
    del models[model_id]
    return RemoveResponse(message=f"Model '{model_id}' removed")


@router.delete("/remove_all", response_model=RemoveResponse, summary="Remove All",
               description="Удаляет все модели из хранилища.")
async def remove_all():
    """
    Удаление всех моделей.
    - Очищаем словарь models.
    - Сбрасываем current_model_id, если была загружена какая-либо модель.
    """
    global current_model_id
    models.clear()
    current_model_id = None
    return RemoveResponse(message="All models removed")
