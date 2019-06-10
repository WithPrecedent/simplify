
from dataclasses import dataclass
import os

@dataclass
class Filer(object):
    """Creates and stores dynamic and static file paths for the siMpLify
    package.
    """
    settings : object
    root : str = '..'
    data : str = 'data'
    results : str = 'results'
    recipes : str = 'dynamic'
    import_file : str = 'data'
    export_file : str = 'data'
    import_format : str = 'csv'
    export_format : str = 'csv'
    use_defaults : bool = True

    def __post_init__(self):
        """Localizes 'files' settings as Filer attributes and sets paths and
        folders.
        """
        self.settings.localize(instance = self, sections = ['files'])
        self._make_folders()
        self._make_io_paths()
        return self

    def _file_name(self, prefix = '', file_name = '', suffix = '',
                   file_type = ''):
        """Creates file name with prefix, suffix, and file extension."""
        extensions = {'csv' : '.csv',
                      'pickle' : '.pkl',
                      'feather' : '.ftr',
                      'h5' : '.hdf',
                      'excel' : '.xlsx',
                      'text' : '.txt',
                      'xml' : '.xml',
                      'png' : '.png'}
        if file_type in extensions:
            return prefix + file_name + suffix + extensions[file_type]
        else:
            return prefix + file_name + suffix + '.' + file_type

    def _iter_path(self, model, recipe_number, splicer = '', file_name = '',
                   file_type = ''):
        """Creates file or folder path for results and recipe exports."""
        if splicer != 'none':
            subfolder = (
                model.technique + '_' + splicer.technique + str(recipe_number))
        else:
            subfolder = model.technique + str(recipe_number)
        self._make_folder(folder = self.make_path(folder = self.recipes,
                                                  subfolder = subfolder))
        return self.make_path(folder = self.recipes,
                              subfolder = subfolder,
                              file_name = file_name,
                              file_type = file_type)

    def _make_folder(self, folder):
        """Creates folder if it doesn't already exist."""
        if not os.path.exists(folder):
             os.makedirs(folder)
        return self

    def _make_folders(self):
        """Creates data and results folders based upon passed parameters."""
        self.data = os.path.join(self.root, self.data)
        self.results = os.path.join(self.root, self.results)
        self._make_folder(self.data)
        self._make_folder(self.results)
        return self

    def _make_io_paths(self):
        """Creates a single import and export path from passed parameters."""
        self.import_path = self.make_path(folder = self.data,
                                          file_name = self.import_file,
                                          file_type = self.import_format)
        self.export_path = self.make_path(folder = self.data,
                                          file_name = self.export_file,
                                          file_type = self.export_format)
        return self

    def make_path(self, folder = '', subfolder = '', prefix = '',
                  file_name = '', suffix = '', file_type = 'csv'):
        """Creates file and/or folder path."""
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