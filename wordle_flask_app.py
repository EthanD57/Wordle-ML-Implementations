from flask import Flask, request, jsonify
from pathlib import Path
import subprocess
import json
import os
import re

from ML.entropy_maximization_bot import EntropyBot
from Utilities.shared_utils import filter_words
import headless_main
import wordle

app = Flask(__name__)

# Loaded once per worker for the in-process entropy bot (no pattern table needed)
GAME = wordle.Wordle(Path(__file__).parent / "words.txt")
WORD_LIST = GAME.word_list
GUESS_PATTERN = re.compile(r"^[a-z]{5}$")

# Valid models
VALID_MODELS = [
    "entropy_maximization",
    "random_forest_classifier",
    "random_forest_regressor",
    "neural_network_classifier",
    "deep_q_network"
]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway."""
    return jsonify({"status": "healthy"}), 200


@app.route('/play', methods=['GET'])
def play_game():
    """
    Play a Wordle game with specified parameters.
    
    Query parameters:
    - word (optional): The word to guess. If not provided, random word is chosen.
    - model (optional): Model to use. Default: entropy_maximization
    
    Example: GET /play?word=crane&model=entropy_maximization
    """
    try:
        word = request.args.get('word', default=None)
        model = request.args.get('model', default='entropy_maximization', type=str)

        # Validate model
        if model not in VALID_MODELS:
            model = 'entropy_maximization'

        # The entropy bot is light enough to run in-process. The ML models stay in a subprocess
        # so torch/sklearn never sit in the gunicorn workers' memory between requests
        if model == 'entropy_maximization':
            result = headless_main.run_game(GAME, model, word)
            return jsonify(result), 200 if result["success"] else 400

        cmd = ['python', 'headless_main.py', '--model', model]

        if word is not None:
            cmd.extend(['--word', word.lower()])

        # Run the headless bot
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )

        # Check for errors
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": f"Bot execution failed: {result.stderr}"
            }), 500

        #Results parsing
        try:
            game_result = json.loads(result.stdout)
            return jsonify(game_result), 200
        except json.JSONDecodeError:
            return jsonify({
                "success": False,
                "error": f"Invalid output from bot: {result.stdout}"
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Game took too long to complete (30 second timeout)"
        }), 504

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/assist', methods=['POST'])
def assist():
    """
    Suggest the next guess for a game being played elsewhere (e.g. the daily Wordle).

    Stateless: the client sends every turn so far and the bot replays them to rebuild its state.
    Scoring is done on the fly, so the user's guesses don't need to be in the word list.

    Body: {"history": [{"guess": "crane", "score": [0, 1, 0, 2, 0]}, ...]}
          score values: 0 = gray, 1 = yellow, 2 = green

    Returns:
        {success, next_guess, remaining_count, remaining_words, solved}
        remaining_words is only filled in once 20 or fewer words remain.
    """
    body = request.get_json(silent=True) or {}
    history = body.get('history', [])

    if not isinstance(history, list) or len(history) > 6:
        return jsonify({"success": False, "error": "History must be a list of at most 6 turns"}), 400

    turns = []
    for turn in history:
        guess = turn.get('guess') if isinstance(turn, dict) else None
        score = turn.get('score') if isinstance(turn, dict) else None
        if not isinstance(guess, str) or not GUESS_PATTERN.match(guess.lower()):
            return jsonify({"success": False, "error": "Each guess must be 5 letters (a-z)"}), 400
        if (not isinstance(score, list) or len(score) != 5
                or any(type(s) is not int or s not in (0, 1, 2) for s in score)):
            return jsonify({"success": False, "error": "Each score must be 5 values of 0, 1, or 2"}), 400
        turns.append((guess.lower(), score))

    solved_at = next((i for i, (_, score) in enumerate(turns) if score == [2] * 5), None)
    if solved_at is not None:
        if solved_at != len(turns) - 1:
            return jsonify({"success": False, "error": "Turns can't continue after the word is solved"}), 400
        return jsonify({"success": True, "solved": True, "next_guess": None,
                        "remaining_count": 1, "remaining_words": [turns[-1][0]]}), 200

    bot = EntropyBot(WORD_LIST)
    for guess, score in turns:
        filter_words(guess, score, bot.game_state)
    bot.game_state.guess_count = len(turns)

    remaining = bot.game_state.remaining_words
    if not remaining:
        return jsonify({
            "success": False,
            "error": "No words match those colors. Double-check them, or the answer might not be in my word list."
        }), 422

    out_of_guesses = len(turns) == 6
    return jsonify({
        "success": True,
        "solved": False,
        "next_guess": None if out_of_guesses else bot.make_guess(),
        "remaining_count": len(remaining),
        "remaining_words": remaining if len(remaining) <= 20 else []
    }), 200


@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    return jsonify({
        "models": VALID_MODELS
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
