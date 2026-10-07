import json
import sys
from pathlib import Path
from random import choice
import argparse

from Utilities.shared_utils import filter_words, score_guess
import wordle

models = {
    "entropy_maximization": 1,
    "random_forest_classifier": 2,
    "random_forest_regressor": 3,
    "neural_network_classifier": 4,
    "deep_q_network": 5
}


def main():
    """
    Headless Wordle bot runner that outputs JSON for web consumption.
    """
    parser = argparse.ArgumentParser(description="Wordle Bot Runner")
    parser.add_argument('--word', type=str, required=False, default=None, help='The word to guess')
    parser.add_argument('--model', type=str, default='entropy_maximization',
                        help='Which model to use', choices=list(models.keys()))

    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse calls sys.exit on error, we need to catch and return JSON
        print(json.dumps({
            "success": False,
            "error": "Invalid arguments provided"
        }))
        return

    try:
        # Load word list
        word_list_path = Path("words.txt")
        if not word_list_path.exists():
            print(json.dumps({
                "success": False,
                "error": "Word list not found"
            }))
            return

        game = wordle.Wordle(word_list_path)
        print(json.dumps(run_game(game, args.model, args.word)))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))


def run_game(game_instance: wordle.Wordle, model_name: str, word: str | None) -> dict:
    """
    Play one game and build the JSON-ready result. Also called in-process by wordle_flask_app.

    Args:
        game_instance: Wordle instance with word list
        model_name: Key of the models dict
        word: Target word. Random if None or not in the word list

    Returns:
        dict: {success, word, model, guesses, won, num_guesses}
    """
    # Checking the word length is NOT required here because this will NOT allow users
    # to enter their own chosen words. If the word list doesn't contain the word sent
    # by the website, it simply picks a random word. This sanitizes input from the website.
    if word is None or word.lower() not in game_instance.word_list:
        target_word = choice(game_instance.word_list)
    else:
        target_word = word.lower()

    guesses = play_game(game_instance, models[model_name], target_word)

    return {
        "success": True,
        "word": target_word,
        "model": model_name,
        "guesses": guesses,
        "won": guesses[-1]["guess"] == target_word,
        "num_guesses": len(guesses)
    }


def play_game(game_instance: wordle.Wordle, model: int, word: str) -> list:
    """
    Run a single Wordle game.

    Args:
        game_instance: Wordle instance with word list
        model: Model ID (1-5)
        word: Target word to guess

    Returns:
        List of guesses with scores
    """
    bot = initialize_bot(game_instance, model)

    guess_count = 0
    guesses = []

    while guess_count < 6:
        guess = bot.make_guess()
        score = score_guess(word, guess)

        # Format: [guess_word, [score_array]]
        guesses.append({
            "guess": guess,
            "score": score  # [0=wrong, 1=wrong position, 2=correct position]
        })

        if guess == word:
            break

        filter_words(guess, score, bot.game_state)
        guess_count += 1

    return guesses


def initialize_bot(game_instance: wordle.Wordle, model: int = 1):
    """
    Initialize the appropriate bot based on model ID.

    Args:
        game_instance: Wordle instance
        model: Model ID (1-5)

    Returns:
        Initialized bot instance
    """
    # Imported lazily so the entropy bot can run without loading torch/sklearn
    if model == 1:
        from ML import entropy_maximization_bot
        # No pattern table: scores are computed on the fly against the remaining words
        return entropy_maximization_bot.EntropyBot(game_instance.word_list)
    elif model == 2:
        from ML import random_forest_classifier
        bot = random_forest_classifier.RandomForestClassifierModel(game_instance.word_list)
        bot.train()
        return bot
    elif model == 3:
        from ML import random_forest_regressor
        bot = random_forest_regressor.RandomForestRegressorModel(game_instance.word_list)
        bot.train()
        return bot
    elif model == 4:
        from ML import neural_network_classifier
        bot = neural_network_classifier.NeuralNetworkClassifier(game_instance.word_list)
        bot.train()
        return bot
    elif model == 5:
        from ML import deep_q_network
        bot = deep_q_network.DQNBot(game_instance.word_list)
        bot.train()
        return bot
    else:
        raise ValueError(f"Unknown model ID: {model}")


if __name__ == '__main__':
    main()