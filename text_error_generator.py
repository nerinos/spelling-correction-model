import random
from typing import List

class AbstractError:

    def makeError(self, sentence: str) -> str:
        raise NotImplementedError
        

# shuffles words in sentence
# takes sentence='str' and probe_a as chance to swap words
class WordShuffle(AbstractError):

    def __init__(self, prob):
        self.prob = prob

    def makeError(self, sentence: str) -> str:
        word_list = sentence.split()
        if len(word_list) <= 2:
            return sentence
        trash_swap = [round(1000 * self.prob), 1000]

        for i in range(len(word_list)):
            random.seed()
            if random.randint(0, trash_swap[1]) <= trash_swap[0] and len(word_list) > 3:
                rand = random.randint(-1, 1)
                if i + rand >= len(word_list) - 1 or i + rand < 0:
                    rand *= -1
                word_list[i], word_list[i + rand] = word_list[i + rand], word_list[i]

        # print('\n', sentence, '\n', [' '.join(word_list)][0], '\n')
        return [' '.join(word_list)][0]



# shuffle letter in each word of given sentence
# takes probe_a as chance to swap letters for each word
class LetterShuffling(AbstractError):

    def __init__(self, prob):
        self.prob = prob
    
    def makeError(self, sentence: str) -> str:
        word_list = sentence.split()
        trash_swap = [round(1000 * self.prob), 1000]

        result = []
        for word in word_list:
            random.seed()
            res = word
            if random.randint(0, trash_swap[1]) <= trash_swap[0] and len(word) > 2:
                rand = random.randint(0, len(word)-2)
                res = word[:rand] + word[rand+1] + word[rand] + word[rand+2:]
            result.append(res)

        # print('\n', sentence, '\n', [' '.join(result)][0], '\n')
        return ' '.join(result)
    
class Typo(AbstractError):

    def __init__(self, prob, prob_replace, space_delete=True):
        self.prob = prob
        self.prob_replace = prob_replace
        self.space_delete = space_delete

    # finds letter for given letter, basing on letter_map
    # takes char and dict
    def generate(self, source_letter, letter_map):
        try:
            list_ = letter_map[source_letter] + source_letter
        except KeyError:
            return source_letter
        random.seed()
        rand = random.randint(0, len(list_))

        # for deleting letter
        if rand == len(list_):
            return ''
        return list_[rand]


    # creating noise in given sentence (add, replace, delete)
    # takes probe_a as chance to do smth with letter probe_b as chance to add instead of replace
    # gives new sentence as result
    def makeError(self, sentence: str) -> str:
        sentence = sentence
        keyboard_map = {'й': 'цыф', 'ц': 'увыфй', 'у': 'кавыц', 'к': 'епаву', 'е': 'нрпак', 'н': 'горпе', 'г': 'шлорн', 'ш': 'щдлог', 'щ': 'зждлш', 'з': 'хэждщ', 'х': 'зжэъ', 'ъ': 'хэ',
                        'ф': 'йцыя', 'ы': 'цувчяф', 'в': 'касчыу', 'а': 'укепмсв', 'п': 'енрима', 'р': 'нготипе', 'о': 'гшльтрн', 'л': 'шщдбьог', 'д': 'щзжюблш', 'ж': 'щзхэ.юд', 'э': 'ъхзж.',
                        'я': 'фыч', 'ч': 'яывс', 'с': 'чвам', 'м': 'сапи', 'и': 'мпрт', 'т': 'ироь', 'ь': 'толб',  'б': 'ьлдю', 'ю': 'бджэ',}

        trash_letter = [round(1000 * self.prob), 1000]
        trash_add = [round(1000 * self.prob_replace), 1000]

        result_ = ''
        for i in range(len(sentence)):
            if sentence[i] == ' ':
                if self.space_delete and random.randint(0, trash_letter[1]) <= trash_letter[0]:
                    continue
                else:
                    result_ += ' '
                    continue
            if random.randint(0, trash_letter[1]) <= trash_letter[0]:
                temp_1 = random.randint(0, trash_add[1])
                generated = self.generate(sentence[i], keyboard_map)

                if temp_1 <= trash_add[0]:
                    if temp_1 % 2 == 0:
                        result_ += sentence[i] + generated
                    else:
                        result_ += generated + sentence[i]
                else:
                    result_ += generated

            else:
                result_ += sentence[i]

        return result_



class CommonMistake(AbstractError):
    common_mistakes = [{'тся', 'ться'}, {'сч', 'щ'}, {'о', 'а'}, {'ъ', 'ь'}, {'жи', 'жы'}, {'ши', 'шы'}, {'ча', 'чя'}, {'ща', 'щя'}]

    def __init__(self, prob):
        self.prob = prob



    def findAndMakeMistake(self, word: str) -> str:
        mistake_candidates = list()
        for item in self.common_mistakes:
            for mistake in item:
                pos = word.lower().find(mistake)
                if pos != -1:
                    temp = item.copy()
                    temp.remove(mistake)
                    mistake_candidates.append((pos, temp, mistake))
        if not mistake_candidates:
            return word
        (pos, mistakes, initial) = random.choice(mistake_candidates)
        generated = random.sample(mistakes, 1)[0]
        upper_case: bool = word.isupper()
        new_word = word[0:pos] + generated + word[pos + len(initial):]
        if upper_case:
            new_word = new_word.upper()
        return new_word

    # changes word's form in given sentence
    # takes probe_a as chance to change word
    # and mistakes_list as list of common_mistakes
    def makeError(self, sentence: str) -> str:
        word_list = sentence.split()

        trash_change = [round(1000 * self.prob), 1000]

        for i in range(len(word_list)):
            word = word_list[i]
            if random.randint(0, trash_change[1]) <= trash_change[0]:
                word_list[i] = self.findAndMakeMistake(word)
            
        return ' '.join(word_list)


class TextErrorGenerator:
    error_list: List[AbstractError] = []

    def __init__(self, errors: List[AbstractError]) -> None:
        self.error_list = errors

    def generateErrors(self, sentence: str) -> str:
        result = sentence
        for error in self.error_list:
            result = error.makeError(result)
        return result