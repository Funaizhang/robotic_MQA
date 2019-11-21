
"""
Generates the question string, given the template
and filler objects and tables
"""

from nltk.stem import WordNetLemmatizer




class QuestionStringBuilder():
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.articleMap = {'utensil': 'a', 'utensil holder': 'a'}
        self.tables = set()
        self.objects = set()

    """
    Checks if a noun is plural or not; requires nltk
    from https://stackoverflow.com/questions/18911589/how-to-test-whether-a-word-is-in-singular-form-or-not-in-python
    """

    def isPlural(self, word):
        lemma = self.wnl.lemmatize(word, 'n')
        plural = True if word is not lemma else False
        return plural

    """
    Handles auxiliary verbs, articles and plurals
    """

    def prepareString(self, template, obj, table):
        qString = template

        self.objects.add(obj)
        self.tables.add(table)

        obj = ' '.join(obj.split('_'))
        table = ' '.join(table.lower().split('_'))

        if '<AUX>' in qString:
            qString = self.replaceAux(qString, obj)
        if '<ARTICLE>' in qString:
            qString = self.replaceArticle(qString, obj)
        if '<OBJ>' in qString or '<OBJ-plural>' in qString:
            qString = self.replaceObj(qString, obj)
        if '<TABLE>' in qString or '<TABLE-plural>' in qString:
            qString = self.replaceTable(qString, table)
        return qString

    def prepareStringForLogic(self, template, obj1, obj2, table, op):
        qString = template

        self.tables.add(table)
        self.objects.add(obj1)
        self.objects.add(obj2)

        table = ' '.join(table.lower().split('_'))
        obj1 = ' '.join(obj1.split('_'))
        obj2 = ' '.join(obj2.split('_'))

        qString = self.replaceAux(qString, obj1)
        qString = self.replaceArticle(qString, obj1)
        qString = self.replaceArticle(qString, obj2)
        qString = self.replaceObjForLogic(qString, obj1, obj2)
        qString = self.replaceTable(qString, table)
        qString = self.replaceOp(qString, op)
        return qString

    def replaceOp(self, template, op):
        assert '<LOGIC>' in template
        return template.replace('<LOGIC>', op)

    def replaceAux(self, template, obj):
        assert '<AUX>' in template
        if self.isPlural(obj):
            return template.replace('<AUX>', 'are')
        else:
            return template.replace('<AUX>', 'is')

    def replaceArticle(self, template, obj):
        assert '<ARTICLE>' in template
        if self.isPlural(obj):
            return template.replace(' <ARTICLE>', '', 1)
        else:
            if obj in self.articleMap:
                return template.replace('<ARTICLE>', self.articleMap[obj], 1)
            elif obj[0] in ['a', 'e', 'i', 'o', 'u']:
                return template.replace('<ARTICLE>', 'an', 1)
            else:
                return template.replace('<ARTICLE>', 'a', 1)

    def replaceObj(self, template, obj):
        if '<OBJ>' in template:
            return template.replace('<OBJ>', obj, 1)
        elif '<OBJ-plural>' in template:
            if self.isPlural(obj):
                return template.replace('<OBJ-plural>', obj)
            else:
                return template.replace('<OBJ-plural>', obj + 's')

    def replaceObjForLogic(self, template, obj1, obj2):
        template = template.replace('<OBJ1>', obj1)
        return template.replace('<OBJ2>', obj2)

    def replaceTable(self, template, table):
        # assert '<TABLE>' in template
        if '<TABLE>' in template:
            return template.replace('<TABLE>', table)
        elif '<TABLE-plural>' in template:
            return template.replace('<TABLE-plural>', table + 's')


if __name__ == '__main__':
    q_str_builder = QuestionStringBuilder()
    q_string = q_str_builder.prepareString(
        "is there a <OBJ> in the <TABLE>?",
        "sofa",
        "desk",
    )
    q_string_for_logic = q_str_builder.prepareStringForLogic(
        "<AUX> there <ARTICLE> <OBJ1> <LOGIC> <ARTICLE> <OBJ2> in the <TABLE>",
        "ottoman", "chairs", "conference table", "and")
    print(q_string_for_logic)
    print(q_string)
    print(q_str_builder.tables)
    print(q_str_builder.objects)

