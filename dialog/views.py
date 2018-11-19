from django.shortcuts import render
from django.http import HttpResponse
from dialog import EDST_test_web


class TestDialogSystem:
    '''
    this is just for testing
    '''
    def __init__(self):
        self.turn = 0
        self.dialog_history = []

    def update(self, usr):
        if usr == "重来":
            self.dialog_history = []
            return self.dialog_history
        else:
            self.dialog_history.append(("usr:" + usr, self.response(usr)))
            return self.dialog_history

    def response(self, usr):
        return "sys：next"

dialog_system = EDST_test_web.DialogSystem()

def init_action(request):
    return render(request, 'init_page.html',{})

def edit_action(request):
    title = request.POST.get('title', '重来')
    dialog = dialog_system.update(title)
    return render(request, 'edit_page.html', {'data':dialog})
