FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

RUN pip install \
    jupyterlab \
    notebook \
    pandas \
    matplotlib \
    numpy \
    scikit-learn

WORKDIR /workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]