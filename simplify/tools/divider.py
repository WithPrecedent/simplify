
from dataclasses import dataclass
import os
import re

from tools.tool import Tool
from utilities.rematch import ReMatch


@dataclass
class Divider(Tool):

    sections : object = None
    sources : object = None
    options : object = None
    suffixes : object = None
    prefixes : object = None
    grid : object = None
    file_path : str = ''
    file_name : str = ''
    remove_from_source = True

    def __post_init__(self):
        super().__post_init__()
        self.options = {'extract' : self._extract,
                        'split' : self._split}
        return

    def _set_dividers(self):
        if not self.grid:
            if self.file_path:
                df = ReMatch(file_path = self.file_path,
                             reverse_dict = True).expressions
            elif self.file_name:
                self.file_path = os.path.join(self.filer.data, self.file_name)
                df = ReMatch(file_path = self.file_path,
                             reverse_dict = True).expressions
            else:
                error = 'Divider requires file_path, file_name or regexes'
                raise AttributeError(error)
            self.grid = df.todict
        return self

    def extract(self, name, regex):
        if re.search(regex, self.sources[name]):
            matched = re.search(regex, self.sources[name]).group(0)
            if self.remove_from_source:
                self.sources[name].replace(matched, '')
                return matched, self.sources[name]
            else:
                return matched

    def separate(self, name, regex):
        matched_list = re.findall(regex, self.sources[name])
        return matched_list




    def _divide_separate_opinions(self, df, bundle, dividers_table):
        separate_list = []
        concur_list = []
        dissent_list = []
        mixed_list = []
        if re.search(dividers_table['separate_div'],
                     bundle['opinions_breaks']):
            separate_list = re.findall(
                    dividers_table['separate_div'],
                    bundle['opinions_breaks'])
            for i in separate_list:
                if re.search(dividers_table['concur_div'], i):
                    if re.search(dividers_table['mixed_div'], i):
                        mixed_list.append(i)
                    else:
                        concur_list.append(i)
                elif re.search(dividers_table['dissent_div'], i):
                    if re.search(dividers_table['mixed_div'], i):
                        mixed_list.append(i)
                    else:
                        dissent_list.append(i)
            df['separate_concur'] = concur_list + mixed_list
            df['separate_dissent'] = dissent_list + mixed_list
        return df, bundle

    def _general_divider(self, df, bundle, dividers_table):
        if self.source_section == 'header':
            if self.section == 'date':
                if re.search(self.regex, bundle['header']):
                    df[self.section_column] = (
                            re.findall(self.regex, bundle['header']))
            elif re.search(self.regex, bundle['header']):
                df[self.section_column] = (
                        re.search(self.regex, bundle['header']).group(0))
                df[self.section_column] = (
                        df[self.section_column].strip())
                bundle['header'] = bundle['header'].replace(
                        df[self.section_column], '')
                df[self.section_column] = self._no_breaks(
                        df[self.section_column])
        elif self.source_section == 'opinions':
            if re.search(self.regex, bundle['opinions']):
                df[self.section_column] = re.search(self.regex,
                        bundle['opinions']).group(0)
        return df, bundle

    def _separate_header(self, df = None, case_text = None, bundle = None):
        """
        Divides court opinion into header and opinions divider. To avoid
        data extraction errors, it is essential to parse the header and
        opinions separately.
        """
        if re.search(self.cases.dividers_table['op_div'],
                     case_text):
            op_list = re.split(self.cases.dividers_table['op_div'],
                               case_text)
            if len(op_list) > 0:
                bundle['header'] = op_list[0]
                if len(op_list) > 1:
                    bundle['opinions'] = self._no_breaks(''.join(op_list[1:]))
                    bundle['opinions_breaks'] = (''.join(op_list[1:]))
        else:
            bundle['header'] = case_text
            bundle['opinions'] = 'none'
            bundle['opinions_breaks'] = 'none'
        if self.source == 'lexis_nexis':
            bundle['header'] = re.sub(
                    self.cases.dividers_table['lex_pat'], '',
                    bundle['header'])
        return df, bundle


    def add(self, techniques, regexes):
        new_techniques = zip(self._listify(techniques),
                             self._listify(regexes))
        for technique, regex in new_techniques.items():
            self.options.update({technique, regex})
        return self

    def mix(self):
        for name, regex in self.dividers:
            self.options[name](name, regex)
        return self