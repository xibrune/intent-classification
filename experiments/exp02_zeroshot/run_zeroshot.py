"""
Эксперимент №2: Zero-Shot классификация интентов CLINC150 с Flan-T5.

Подход:
  Для каждого примера из test.csv задаём модели вопрос по каждому из 150 интентов.
  Интент с наибольшей вероятностью ответа "yes" считается предсказанием.
  Повторяем для трёх типов промптов и сравниваем результаты.

Промпты:
  1. Короткий:      "classify: {text} class: {intent}"
  2. Описательный:  "classify: {text} class: {intent}. description: {description}"
  3. Вопрос:        "does the user want to {intent}? text: {text}"
"""

import argparse
import os
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import T5ForConditionalGeneration, T5Tokenizer


# Маппинг id -> название интента
ID_TO_INTENT = {
    0: "restaurant_reviews",
    1: "nutrition_info",
    2: "account_blocked",
    3: "oil_change_how",
    4: "time",
    5: "weather",
    6: "redeem_rewards",
    7: "interest_rate",
    8: "gas_type",
    9: "accept_reservations",
    10: "smart_home",
    11: "user_name",
    12: "report_lost_card",
    13: "repeat",
    14: "whisper_mode",
    15: "what_are_your_hobbies",
    16: "order",
    17: "jump_start",
    18: "schedule_meeting",
    19: "meeting_schedule",
    20: "freeze_account",
    21: "what_song",
    22: "meaning_of_life",
    23: "restaurant_reservation",
    24: "traffic",
    25: "make_call",
    26: "text",
    27: "bill_balance",
    28: "improve_credit_score",
    29: "change_language",
    30: "no",
    31: "measurement_conversion",
    32: "timer",
    33: "flip_coin",
    34: "do_you_have_pets",
    35: "balance",
    36: "tell_joke",
    37: "last_maintenance",
    38: "exchange_rate",
    39: "uber",
    40: "car_rental",
    41: "credit_limit",
    42: "oos",
    43: "shopping_list",
    44: "expiration_date",
    45: "routing",
    46: "meal_suggestion",
    47: "tire_change",
    48: "todo_list",
    49: "card_declined",
    50: "rewards_balance",
    51: "change_accent",
    52: "vaccines",
    53: "reminder_update",
    54: "food_last",
    55: "change_ai_name",
    56: "bill_due",
    57: "who_do_you_work_for",
    58: "share_location",
    59: "international_visa",
    60: "calendar",
    61: "translate",
    62: "carry_on",
    63: "book_flight",
    64: "insurance_change",
    65: "todo_list_update",
    66: "timezone",
    67: "cancel_reservation",
    68: "transactions",
    69: "credit_score",
    70: "report_fraud",
    71: "spending_history",
    72: "directions",
    73: "spelling",
    74: "insurance",
    75: "what_is_your_name",
    76: "reminder",
    77: "where_are_you_from",
    78: "distance",
    79: "payday",
    80: "flight_status",
    81: "find_phone",
    82: "greeting",
    83: "alarm",
    84: "order_status",
    85: "confirm_reservation",
    86: "cook_time",
    87: "damaged_card",
    88: "reset_settings",
    89: "pin_change",
    90: "replacement_card_duration",
    91: "new_card",
    92: "roll_dice",
    93: "income",
    94: "taxes",
    95: "date",
    96: "who_made_you",
    97: "pto_request",
    98: "tire_pressure",
    99: "how_old_are_you",
    100: "rollover_401k",
    101: "pto_request_status",
    102: "how_busy",
    103: "application_status",
    104: "recipe",
    105: "calendar_update",
    106: "play_music",
    107: "yes",
    108: "direct_deposit",
    109: "credit_limit_change",
    110: "gas",
    111: "pay_bill",
    112: "ingredients_list",
    113: "lost_luggage",
    114: "goodbye",
    115: "what_can_i_ask_you",
    116: "book_hotel",
    117: "are_you_a_bot",
    118: "next_song",
    119: "change_speed",
    120: "plug_type",
    121: "maybe",
    122: "w2",
    123: "oil_change_when",
    124: "thank_you",
    125: "shopping_list_update",
    126: "pto_balance",
    127: "order_checks",
    128: "travel_alert",
    129: "fun_fact",
    130: "sync_device",
    131: "schedule_maintenance",
    132: "apr",
    133: "transfer",
    134: "ingredient_substitution",
    135: "calories",
    136: "current_location",
    137: "international_fees",
    138: "calculator",
    139: "definition",
    140: "next_holiday",
    141: "update_playlist",
    142: "mpg",
    143: "min_payment",
    144: "change_user_name",
    145: "restaurant_suggestion",
    146: "travel_notification",
    147: "cancel",
    148: "pto_used",
    149: "travel_suggestion",
    150: "change_volume",
}


# Описания интентов для описательного промпта
INTENT_DESCRIPTIONS = {
    "restaurant_reviews":       "The user wants to find or read reviews for a restaurant.",
    "nutrition_info":           "The user wants nutritional information about a food item or meal.",
    "account_blocked":          "The user's account has been blocked and they need help unblocking it.",
    "oil_change_how":           "The user wants to know how to perform an oil change on their vehicle.",
    "time":                     "The user wants to know the current time.",
    "weather":                  "The user wants to know the current weather or a weather forecast.",
    "redeem_rewards":           "The user wants to redeem loyalty or cashback rewards from their account.",
    "interest_rate":            "The user wants to know the interest rate on a loan, credit card, or savings account.",
    "gas_type":                 "The user wants to know what type of gasoline their vehicle requires.",
    "accept_reservations":      "The user wants to know if a restaurant accepts reservations.",
    "smart_home":               "The user wants to control or get information about a smart home device.",
    "user_name":                "The user wants to find, set, or change their account username.",
    "report_lost_card":         "The user wants to report a lost or stolen credit or debit card.",
    "repeat":                   "The user wants the assistant to repeat its last response or action.",
    "whisper_mode":             "The user wants to enable a quiet or whisper mode on their voice assistant.",
    "what_are_your_hobbies":    "The user is asking the assistant about its hobbies or personal interests.",
    "order":                    "The user wants to place an order for a product or service.",
    "jump_start":               "The user wants to know how to jump start a car with a dead battery.",
    "schedule_meeting":         "The user wants to schedule a meeting with one or more people.",
    "meeting_schedule":         "The user wants to check their upcoming meeting schedule or agenda.",
    "freeze_account":           "The user wants to freeze or lock their bank account for security.",
    "what_song":                "The user wants to know the name of the song that is currently playing.",
    "meaning_of_life":          "The user is asking a philosophical question about the meaning or purpose of life.",
    "restaurant_reservation":   "The user wants to make or manage a reservation at a restaurant.",
    "traffic":                  "The user wants to know about current traffic conditions on a route.",
    "make_call":                "The user wants to place a phone call to a contact.",
    "text":                     "The user wants to send a text message to someone.",
    "bill_balance":             "The user wants to check the remaining balance or amount due on a bill.",
    "improve_credit_score":     "The user wants advice on how to improve their credit score.",
    "change_language":          "The user wants to change the language setting of an application or device.",
    "no":                       "The user is declining or disagreeing with a previous statement.",
    "measurement_conversion":   "The user wants to convert a value between different units of measurement.",
    "timer":                    "The user wants to set a countdown timer for a specific duration.",
    "flip_coin":                "The user wants to flip a virtual coin to get a random heads or tails result.",
    "do_you_have_pets":         "The user is asking the assistant if it has or likes pets.",
    "balance":                  "The user wants to check their bank account balance or available funds.",
    "tell_joke":                "The user wants the assistant to tell them a joke.",
    "last_maintenance":         "The user wants to know when their vehicle last had a maintenance service.",
    "exchange_rate":            "The user wants to know the currency exchange rate between two countries.",
    "uber":                     "The user wants to request a ride through Uber or a similar rideshare service.",
    "car_rental":               "The user wants to rent a car or get information about car rental options.",
    "credit_limit":             "The user wants to know their current credit card spending limit.",
    "oos":                      "The user's request does not match any of the supported intents.",
    "shopping_list":            "The user wants to create or view their grocery or shopping list.",
    "expiration_date":          "The user wants to know the expiration date on their credit or debit card.",
    "routing":                  "The user wants to find their bank's routing number.",
    "meal_suggestion":          "The user wants suggestions for what to eat or cook.",
    "tire_change":              "The user wants to know how to change a flat tire on their vehicle.",
    "todo_list":                "The user wants to create or view their personal to-do list.",
    "card_declined":            "The user's payment card was declined and they want to know why.",
    "rewards_balance":          "The user wants to check how many loyalty or rewards points they have.",
    "change_accent":            "The user wants to change the voice accent of the assistant.",
    "vaccines":                 "The user wants to know what vaccines are required or recommended for travel.",
    "reminder_update":          "The user wants to change or delete an existing reminder.",
    "food_last":                "The user wants to know how long a specific food item will stay fresh.",
    "change_ai_name":           "The user wants to change the name or alias of the AI assistant.",
    "bill_due":                 "The user wants to know the due date for an upcoming bill payment.",
    "who_do_you_work_for":      "The user wants to know which company or organization the assistant works for.",
    "share_location":           "The user wants to share their current location with another person.",
    "international_visa":       "The user wants information about obtaining a visa for international travel.",
    "calendar":                 "The user wants to view events or manage their personal calendar.",
    "translate":                "The user wants to translate a word or phrase into another language.",
    "carry_on":                 "The user wants to know the carry-on baggage rules or restrictions for a flight.",
    "book_flight":              "The user wants to search for or book a flight to a destination.",
    "insurance_change":         "The user wants to update or change their insurance policy or coverage.",
    "todo_list_update":         "The user wants to add, remove, or modify items on their to-do list.",
    "timezone":                 "The user wants to know the time zone of a specific location or city.",
    "cancel_reservation":       "The user wants to cancel an existing reservation at a restaurant or hotel.",
    "transactions":             "The user wants to view or check their recent account transactions.",
    "credit_score":             "The user wants to check or learn about their credit score.",
    "report_fraud":             "The user wants to report fraudulent or suspicious activity on their account.",
    "spending_history":         "The user wants to review their spending history or past financial transactions.",
    "directions":               "The user wants directions or navigation instructions to a destination.",
    "spelling":                 "The user wants to know how to correctly spell a word.",
    "insurance":                "The user wants information about insurance policies, coverage, or claims.",
    "what_is_your_name":        "The user wants to know the name of the AI assistant.",
    "reminder":                 "The user wants to set a reminder for a future task or event.",
    "where_are_you_from":       "The user wants to know the origin or background of the assistant.",
    "distance":                 "The user wants to know the distance between two locations.",
    "payday":                   "The user wants to know when their next paycheck or payday is.",
    "flight_status":            "The user wants to check the real-time status of a specific flight.",
    "find_phone":               "The user wants to locate or find their missing phone.",
    "greeting":                 "The user is greeting or saying hello to the assistant.",
    "alarm":                    "The user wants to set, check, modify, or cancel an alarm.",
    "order_status":             "The user wants to check the current status of an order they placed.",
    "confirm_reservation":      "The user wants to confirm an existing reservation at a restaurant or hotel.",
    "cook_time":                "The user wants to know how long it takes to cook a specific food item.",
    "damaged_card":             "The user has a damaged bank card and wants to know what to do.",
    "reset_settings":           "The user wants to reset a device or application to its default settings.",
    "pin_change":               "The user wants to change the PIN for their bank or debit card.",
    "replacement_card_duration":"The user wants to know how long it takes to receive a replacement card.",
    "new_card":                 "The user wants to apply for a new credit or debit card.",
    "roll_dice":                "The user wants to roll a virtual die or set of dice.",
    "income":                   "The user wants information about their income, earnings, or salary.",
    "taxes":                    "The user wants information or help with filing or understanding taxes.",
    "date":                     "The user wants to know today's date or the date of a specific event.",
    "who_made_you":             "The user wants to know who created or developed the AI assistant.",
    "pto_request":              "The user wants to request paid time off, vacation days, or sick leave.",
    "tire_pressure":            "The user wants to know the correct tire pressure for their vehicle.",
    "how_old_are_you":          "The user wants to know how old the AI assistant is.",
    "rollover_401k":            "The user wants information about rolling over a 401k retirement account.",
    "pto_request_status":       "The user wants to check the status of a previously submitted PTO request.",
    "how_busy":                 "The user wants to know how busy a place is at a specific time.",
    "application_status":       "The user wants to check the status of a credit card or loan application.",
    "recipe":                   "The user wants a recipe for a specific dish or meal.",
    "calendar_update":          "The user wants to add, edit, or delete an event on their calendar.",
    "play_music":               "The user wants to play a specific song, artist, album, or playlist.",
    "yes":                      "The user is affirming or agreeing with a previous statement or question.",
    "direct_deposit":           "The user wants to set up or get information about direct deposit for their paycheck.",
    "credit_limit_change":      "The user wants to request an increase or decrease in their credit card limit.",
    "gas":                      "The user wants to find a nearby gas station.",
    "pay_bill":                 "The user wants to pay a bill or schedule a bill payment.",
    "ingredients_list":         "The user wants a list of ingredients needed for a specific recipe.",
    "lost_luggage":             "The user wants to report lost luggage or get information about a lost bag.",
    "goodbye":                  "The user is ending the conversation or saying goodbye to the assistant.",
    "what_can_i_ask_you":       "The user wants to know what questions or tasks the assistant can help with.",
    "book_hotel":               "The user wants to search for or book a hotel room.",
    "are_you_a_bot":            "The user wants to know if they are talking to a bot or artificial intelligence.",
    "next_song":                "The user wants to skip to the next song in the current playlist.",
    "change_speed":             "The user wants to change the playback speed of audio or video.",
    "plug_type":                "The user wants to know what type of electrical plug is used in a country.",
    "maybe":                    "The user is expressing uncertainty or giving an indeterminate answer.",
    "w2":                       "The user wants information about their W-2 wage and tax statement form.",
    "oil_change_when":          "The user wants to know when their vehicle is next due for an oil change.",
    "thank_you":                "The user is expressing gratitude or saying thank you to the assistant.",
    "shopping_list_update":     "The user wants to add or remove items from their shopping list.",
    "pto_balance":              "The user wants to know how many paid time off days they have remaining.",
    "order_checks":             "The user wants to order paper checks or a new checkbook from their bank.",
    "travel_alert":             "The user wants to know about travel alerts or warnings for a destination.",
    "fun_fact":                 "The user wants to hear an interesting or surprising fun fact.",
    "sync_device":              "The user wants to synchronize their device with another device or account.",
    "schedule_maintenance":     "The user wants to schedule a maintenance appointment for their vehicle.",
    "apr":                      "The user wants to know the annual percentage rate on a credit card or loan.",
    "transfer":                 "The user wants to transfer money between bank accounts or to another person.",
    "ingredient_substitution":  "The user needs to find a substitute for a cooking ingredient.",
    "calories":                 "The user wants to know the calorie count of a specific food or meal.",
    "current_location":         "The user wants to know their current GPS location or address.",
    "international_fees":       "The user wants to know about fees charged for international transactions.",
    "calculator":               "The user wants to perform a mathematical calculation.",
    "definition":               "The user wants the dictionary definition of a word or phrase.",
    "next_holiday":             "The user wants to know when the next public or bank holiday is.",
    "update_playlist":          "The user wants to add or remove songs from a music playlist.",
    "mpg":                      "The user wants to know the fuel efficiency in miles per gallon of a vehicle.",
    "min_payment":              "The user wants to know the minimum payment due on a credit card or loan.",
    "change_user_name":         "The user wants to change their display name or username on an account.",
    "restaurant_suggestion":    "The user wants a recommendation or suggestion for a restaurant to visit.",
    "travel_notification":      "The user wants to notify their bank about upcoming travel to avoid card blocks.",
    "cancel":                   "The user wants to cancel a previous request, action, or subscription.",
    "pto_used":                 "The user wants to know how many paid time off days they have already used.",
    "travel_suggestion":        "The user wants recommendations or suggestions for a travel destination.",
    "change_volume":            "The user wants to adjust the volume level on their device or media player.",
}


# Построение промптов
def make_prompt(prompt_type, text, intent):
    """
    prompt_type=1  Короткий: "classify: {text} class: {intent}"
    prompt_type=2  Описательный: "classify: {text} class: {intent}. description: {desc}"
    prompt_type=3  Вопрос: "does the user want to {intent}? text: {text}"
    """
    if prompt_type == 1:
        intent_label = intent.replace("_", " ")
        return f"classify: {text} class: {intent_label}"

    elif prompt_type == 2:
        intent_label = intent.replace("_", " ")
        desc = INTENT_DESCRIPTIONS[intent]
        return f'classify: {text} class: "{intent_label}". description: {desc}'

    elif prompt_type == 3:
        intent_label = intent.replace("_", " ")
        return f"does the user want to {intent_label}? text: {text}"

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


# Оценка одного примера с батчингом по интентам
def predict_intent(
    text,
    intents,
    prompt_type,
    model,
    tokenizer,
    device,
    batch_size=32,
):
    """
    Для одного текста формирует батч промптов по всем интентам
    и за несколько forward pass'ов выбирает лучший интент.
    """
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    scores = []

    # Делим 151 интент на батчи по batch_size штук
    for batch_start in range(0, len(intents), batch_size):
        batch_intents = intents[batch_start: batch_start + batch_size]

        # Формируем промпты для каждого интента в батче
        prompts = [make_prompt(prompt_type, text, intent) for intent in batch_intents]

        # Токенизируем весь батч сразу
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        # Декодер начинает с токена <pad> для каждого примера в батче
        decoder_input_ids = torch.full(
            (len(batch_intents), 1),
            tokenizer.pad_token_id,
            dtype=torch.long,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids,
            )

        # logits shape: (batch_size, 1, vocab_size)
        batch_scores = outputs.logits[:, 0, yes_token_id].tolist()
        scores.extend(batch_scores)

    # Интент с максимальным score — победитель
    best_idx = scores.index(max(scores))
    return intents[best_idx]


# Основная функция
def run(data_path: str, results_dir: str):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Устройство: {device}")

    # Загрузка данных
    df = pd.read_csv(data_path)
    print(f"Количество примеров: {len(df)}")

    # Преобразуем числовой id интента в строковое название
    df["intent_name"] = df["intent"].map(ID_TO_INTENT)

    # Проверка — все id известны?
    unknown = df["intent_name"].isna().sum()
    if unknown > 0:
        print(f"{unknown} примеров имеют неизвестный ID")
        df = df.dropna(subset=["intent_name"])

    intents = list(ID_TO_INTENT.values())  # все 151 интент
    print(f"Всего интентов: {len(intents)}")

    # Модель
    model_name = "google/flan-t5-base"
    print(f"\nЗагрузка модели: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print("Модель загружена.")

    # Прогон трёх промптов
    os.makedirs(results_dir, exist_ok=True)
    all_results = []
    summary_rows = []

    for prompt_type in [1, 2, 3]:
        prompt_names = {1: "короткий", 2: "описательный", 3: "вопрос"}
        prompt_name = prompt_names[prompt_type]
        print(f"\n{'='*60}")
        print(f"Промпт {prompt_type}: {prompt_name}")
        print(f"{'='*60}")

        predictions = []
        start_time = time.time()

        for i, row in enumerate(df.itertuples(), 1):
            pred = predict_intent(
                text=row.text,
                intents=intents,
                prompt_type=prompt_type,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            predictions.append(pred)

            # Прогресс каждые 50 примеров
            if i % 50 == 0 or i == len(df):
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (len(df) - i)
                print(f"  [{i}/{len(df)}]  elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s")

        # Метрики
        y_true = df["intent_name"].tolist()
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average="macro", zero_division=0)

        print(f"\nПромпт {prompt_type} ({prompt_name}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {f1:.4f}")

        # Сохраняем предсказания
        for text, true, pred in zip(df["text"], y_true, predictions):
            all_results.append({
                "text": text,
                "true_intent": true,
                f"pred_prompt{prompt_type}": pred,
                f"correct_prompt{prompt_type}": int(true == pred),
            })

        summary_rows.append({
            "prompt_id": prompt_type,
            "prompt_name": prompt_name,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(f1, 4),
        })

    # Сохранение результатов

    # Подробные предсказания — объединяем по тексту
    results_df = pd.DataFrame({"text": df["text"].tolist(), "true_intent": df["intent_name"].tolist()})
    for prompt_type in [1, 2, 3]:
        col_pred = f"pred_prompt{prompt_type}"
        col_correct = f"correct_prompt{prompt_type}"
        preds = [r[col_pred] for r in all_results if col_pred in r]
        corrects = [r[col_correct] for r in all_results if col_correct in r]
        results_df[col_pred] = preds
        results_df[col_correct] = corrects

    results_path = os.path.join(results_dir, "exp02_zeroshot_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nРезультаты сохранены: {results_path}")

    # Таблица метрик
    metrics_df = pd.DataFrame(summary_rows)
    metrics_path = os.path.join(results_dir, "exp02_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Метрики сохранены: {metrics_path}")

    # Итоговый вывод
    print("\n" + "="*60)
    print("ИТОГИ")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot intent classification with Flan-T5")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/splits/test.csv"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/metrics"
    )
    args = parser.parse_args()
    run(args.data_path, args.results_dir)