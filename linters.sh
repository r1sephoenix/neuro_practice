#!/bin/bash

black bci_classifier
isort bci_classifier
flake8 bci_classifier
