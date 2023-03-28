class EmojiVocab:

    emoji_dict: dict = {
        'happy': '😀',
        'sad': '😢',
        'angry': '😡',
        'surprised': '😮',
        'disgusted': '🤢',
        'fearful': '😨',
        'neutral': '😐',
        'calm': '😌',
        'confused': '😕',
        'sleepy': '😴',
        'annoyed': '😒',
        'amused': '😆',
        'afraid': '😨',
        'amazed': '😲',
        'ashamed': '😳',
        'attracted': '😍',
        'bad': '😞',
        'cold': '🥶',
        'crazy': '🤪',
        'creepy': '👻',
        'depressed': '😔',
        'devastated': '😭',
        'embarrassed': '😳',
        'frustrated': '😤',
        'grumpy': '😠',
        'guilty': '😔',
        'hopeful': '🤞',
        'hot': '🥵',
        'hungry': '🤤',
        'hurt': '😣',
        'impressed': '😮',
        'jealous': '😒',
        'lonely': '😔',
        'nervous': '😬',
        'proud': '😌',
        'scared': '😨',
        'sick': '🤒',
        'stressed': '😓',
        'surprised': '😮',
        'tired': '😴',
        'worried': '😟',
        'good': '😊',
        'great': '😊',
        'happy': '😊',
        'hilarious': '😆',
        'honest': '😊',
        'humorous': '😆',
        'kind': '😊',
        'legendary': '😎',
        'mischievous': '😈',
        'mysterious': '🤔',
        'mean': '😡',
        'miserable': '😞',
        'mad': '😡',
        'motivated': '😎',
        'nerdy': '🤓',
        'normal': '😐',
        'obnoxious': '😒',
        'optimistic': '😊',
        'oops': '😳',
        'outraged': '😡',
        'peaceful': '😌',
        'playful': '😈',
        'puzzled': '🤔',
        'quirky': '🤪',
        'pleasant': '😊',
        'relaxed': '😌',
        'relieved': '😌',
        'repulsive': '🤮',
        'romantic': '😍',
        'rude': '😒',
        'silly': '🤪',
        'random': '🤪',
        'realistic': '😐',
        'satisfied': '😌',
        'serious': '😐',
        'shocked': '😮',
        'shy': '😳',
        'silent': '😐',
        'tense': '😬',
        'tough': '😐',
        'uncomfortable': '😐',
    }

    vocab: list = list(emoji_dict.keys())
    emoji_dict_unicode: dict = {k: v.encode('unicode-escape').decode('utf-8') for k, v in emoji_dict.items()}

class GestureVocab:

    gesture_dict = {
        'thumbs up': '👍',
        'thumbs down': '👎',
        'ok': '👌',
        'fist': '✊',
        'victory': '✌️',
        'peace': '✌️',
        'rock': '🤘',
        'metal': '🤘',
        'vulcan': '🖖',
        'hand': '✋',
        'raised hand': '✋',
        'raised hands': '🙌',
        'clap': '👏',
        'wave': '👋',
        'call me': '🤙',
        'call me hand': '🤙',
    }

    vocab: list = list(gesture_dict.keys())
    gesture_dict_unicode: dict = {k: v.encode('unicode-escape').decode('utf-8') for k, v in gesture_dict.items()}