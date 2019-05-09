"""
Class and methods used by siMpLify to store data and results in consistent
path structure based upon user options.
"""
from dataclasses import dataclass
import os

@dataclass
class Filer(object):
    """
    Class for dynamically creating file paths.
    """
    settings : object = None
    root : str = ''
    data : str = ''
    results : str = ''
    recipe_folder : str = ''
    import_file : str = ''
    export_file : str = ''
    import_format : str = ''
    export_format : str = ''
    use_defaults : bool = True

    def __post_init__(self):
        if self.settings:
            self.settings.simplify(class_instance = self, sections = ['files'])
        if self.use_defaults:
            if not self.root:
                self.root = '..'
            if not self.data:
                self.data = os.path.join(self.root, 'data')
            if not self.results:
                self.results = os.path.join(self.root, 'results')
            if not self.import_format:
                self.import_format = 'csv'
            if not self.export_format:
                self.export_format = 'csv'
            if not self.recipe_folder:
                self.recipe_folder = 'dynamic'
        self._make_folder(self.data)
        self._make_folder(self.results)
        self._make_io_paths()
        return self

    def _make_io_paths(self):
        self.import_path = self.make_path(folder = self.data,
                                          file_name = self.import_file,
                                          file_type = self.import_format)
        self.export_path = self.make_path(folder = self.data,
                                          file_name = self.export_file,
                                          file_type = self.export_format)
        return self

    def _file_name(self, prefix = '', file_name = '', suffix = '',
                   file_type = ''):
        extensions = {'csv' : '.csv',
                      'pickle' : '.pkl',
                      'feather' : '.ftr',
                      'h5' : '.hdf',
                      'excel' : '.xlsx',
                      'text' : '.txt',
                      'xml' : '.xml',
                      'png' : '.png'}
        return prefix + name + suffix + extensions[file_type]

    def _make_folder(self, folder):
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _iter_path(self, model, recipe_number, splicer = '', file_name = '',
                   file_type = ''):
        if splicer:
            subfolder = model.name + '_' + splicer.name + recipe_number
        else:
            subfolder = model.name + recipe_number
        self._make_folder(folder = self.make_path(
                folder = self.recipes,
                subfolder = subfolder))
        return self.make_path(folder = self.recipes,
                              subfolder = subfolder,
                              file_name = file_name,
                              file_type = file_type)


    def make_path(self, folder = '', subfolder = '', prefix = '',
                  file_name = '', suffix = '', file_type = 'csv'):
        if subfolder:
            folder = os.path.join(folder, subfolder)
        if file_name:
            file_name = self._file_name(prefix = prefix,
                                        file_name = file_name,
                                        suffix = suffix,
                                        file_type = file_type)
            return os.path.join(folder, file_name)
        else:
            return folder