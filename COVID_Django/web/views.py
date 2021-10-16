from rest_framework import generics, serializers, exceptions
from rest_framework.response import Response
from django.shortcuts import render

from os.path import join as join_path

from .core import Core

core = Core()

class PredictSerializer(serializers.Serializer):
	files = serializers.FileField()

class HealthCheck(generics.GenericAPIView):
	def get(self, request, *args, **kwargs):
		return Response({"status": "OK"})

def home(request):
	return render(request, 'index.html')

class PredictAPI(generics.GenericAPIView):
	serializer_class = PredictSerializer
	authentication_classes = []
	def post(self, request, *args, **kwargs):
		serializer = self.get_serializer(data=request.data)
		if serializer.is_valid():
			result = core.work(join_path("frontend", "temp", "def.png"), serializer.validated_data['files'])
			print(result)
			return Response(result)
		else: 
			return Response({"error": serializer.errors})


class DicomPredictAPI(generics.GenericAPIView):
	serializer_class = PredictSerializer
	authentication_classes = []
	def post(self, request, *args, **kwargs):
		serializer = self.get_serializer(data=request.data)
		if serializer.is_valid():
			path = join_path("frontend", "temp")
			result = core.work_dicom(path, serializer.validated_data['files'])
			print(result)
			return Response(result)
		else: 
			return Response({"error": serializer.errors})