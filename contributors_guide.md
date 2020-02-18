# Contribution Guidelines

siMpLify uses a code structure patterned after the writing process. Each major subpackage in the siMpLify package  (Wrangler, Analyst, Explorer, Critic, Artist)creates a Book object which contains particular implementations (Chapters) which have one or more steps (Repository).

    Wrangler creates an Manual of Plans.
    Analyst creates a Cookbook of Recipes.
    Explorer creates a Ledger of Summaries.
    Critic creates a Collection of Reviews.
    Artist creates a Canvas of Illustrations.

siMpLify is fully extensible. Additional subpackages, Books, Chapters, and Repository can be added to a Project. To contribute to siMpLify, please follow these basic rules:

## Style

1. The project generally follows the Google Style Guide for python:
    https://google.github.io/styleguide/pyguide.html
It is particularly important for contributions to follow the Google style for docstrings so that sphinx napoleon can automatically incorporate the docstrings into online documentation.

2. Explicitness preferences are heightened beyond PEP8 guidelines. Varible names should be verbose enough so that there meaning is clear and consistent. Annotations (using python 3.7+) should always be used in arguments and docstrings. As siMpLify is intended to be used by all levels of coders (and by non-coders as well), it is important to make everything as clear as possible to someone seeing the code for the first time. List and dict comprehensions are disfavored. If there are significant speed advantages to using a comprehension,
please wrap them in a function or method (as with the 'add_suffix' and 'add_prefix' functions in simplify.core.utilities).

3. Follow the package naming conventions. All abstract base classes begin with the prefix 'Simple'. Generally, siMpLify tries to avoid cluttering user namespace with commonly used object names (an exception was made for the 'apply' method).

4. siMpLify follows an object-oriented approach because that makes integration with scikit-learn and general modularity easier. Contributions are not precluded from using other programming styles, but class wrappers might be needed to interface properly with the overall siMpLify structure. In fact, the interfaces for deep learning packages are largely wrappers for functional programming.

## Structure

1. All base classes should have a similar interface of methods. Each base class divides processes into three stages, again patterned after the writing process which are the core methods used throughout the siMpLify package:

    * draft: sets default attributes (required).
    * publish: finalizes attributes after any runtime changes. (required).
    * apply: applies selected options to passed arguments (optional).

Any new subpackages, Books, Chapters, and Repository should follow a similar template. All classes within siMpLify should use the new @dataclass accessor to minimize boilerplate code (introduced in python 3.7)

2. siMpLify lazily (runtime) loads most external and internal modules. This is done to lower overhead and incorporate "soft" dependencies. As a result, contributions hould follow these general idioms for importing objects within modules.

    For Book-level classes, all potentially importable objects should be stored in a dict called 'options'. Each entry in 'options' should follow this format:

        {key(str): (module_path(str), object_name(str))}

    Then, to import the needed object, use this general code:

        from importlib import import_module

        getattr(import_module(self.workers[key][0]), self.workers[key][1])

    For Technique-level classes, a special class has been created to construct needed external and internal objects. It is the Option class in the Contributor module. Follow the documentation there for creating Repository.

    Chapters should not require an module importation.

3. siMpLify favors coomposition over inheritance and makes extensive use of the composite and builder design patterns. Inheritance is used, and only allowed from the abstract base classes that define a particular grouping of classes. For example, the Book, Chapter, and Technique classes inherit from Manuscript to allow for sharing of common methods.

4. When composing objects through a loosely coupled hierarchy, it is important to provide connections in both directions. For example, the Chapter class has methods to 'add_technique' and 'add_book' which automatically change local attributes ('techniques' and 'book') accordingly. This is done so that any class in a composite tree can access attributes from other classes in that tree without passing numerous arguments.

## siMpLify Worker

1. All file management should be perfomed throught the shared Inventory instance.

2. All external settings should be imported and constructed using the shared Idea instance. To inject matching attributes from the Idea instance, use this idiom from a subclass with the Idea instance stored at 'idea':

    self = self.idea.apply(instance = self)

3. All external data should be contained in instances of Dataset. Before beginning the processes in Analyst, ideally, there should be a single, combined pandas DataFrame stored in the Dataset instance at the 'df' attribute.

4. Any generally usable functions or decorators should be stored in simplify.core.utilities.

5. If you create a proxy for typing, please subclass the SimpleType class in simplify.core.definitionsetter, if possible.

6. State management is currently handled by classes in simplify.core.states, but are typically accessed indirectly. The overall 'worker' attribute is an attribute to a Inventory instance and 'data_state' is an attribute to an Dataset instance.

## General

1. When in doubt, copy. All of the core subpackages follow these rules. If you are starting a new object in siMpLify, the safest route is just to copy an analagous class (and related import statements) into a new module and go from there.

2. Add any new soft or hard dependencies to the requirements.txt and yaml environment files in the root folder of the package. Even though there is a risk to the approach, siMpLify favors importation over integration of open-source code. This allows updates to those external dependencies to be seamlessly added into a siMpLify workflow. This can create problems when constructing virtual python environments, but, absent special circumstances, importatiion is preferred.

3. If you have a great idea that is inconsistent with these guidelines, email Corey Yung directly. We are always looking for ways to improve siMpLify and are open to amending or discarding various contribution guidelines if they are stifling innovation.