from django.shortcuts import render
from beer.models import *
from django.contrib.auth.decorators import login_required
from cart.cart import Cart
# Create your views here.


@login_required(login_url="/user/login")
def cart_add(request, id):
    cart = Cart(request)
    product = Product.objects.get(id=id)
    cart.add(product=product)
    return redirect("/")


@login_required(login_url="/user/login")
def item_clear(request, id):
    cart = Cart(request)
    product = Product.objects.get(id=id)
    cart.remove(product)
    return redirect("cart_detail")


@login_required(login_url="/user/login")
def cart_clear(request):
    cart = Cart(request)
    cart.clear()
    return redirect("cart_detail")


@login_required(login_url="/user/login")
def cart_detail(request):
    return render(request, 'cart/cart_detail.html')
