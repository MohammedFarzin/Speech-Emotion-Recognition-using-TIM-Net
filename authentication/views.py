from django.shortcuts import render, redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required

# Create your views here.
def signin(request):
    if request.user.is_authenticated:
        print('autheticated')
        return redirect('home')
    else:
        print('Iam here')
        if request.method == 'POST':
            username = request.POST['username']
            password1 = request.POST['password1']


            user = authenticate(username = username, password = password1)
            print('authenticating')
            if user is not None:
                login(request, user)
                print('login')
                return redirect('home')

            else:
                print('not login')
                messages.error(request, 'Bad credentials')
                return redirect('home')
    return render(request, 'signin.html')


@login_required(login_url='signin')
def signout(request):
    print("this is log")

    logout(request)
    print("this is logout")
    messages.success(request, 'Logout successful')
    return redirect('signin')