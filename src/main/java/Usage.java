import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class Usage {

	public static void main(String[] args) throws IOException, MalformedModelException, TranslateException {
		Image img = ImageFactory.getInstance().fromUrl("https://storage.googleapis.com/kagglesdsdata/datasets/2265389/3799619/mnist_png/test/8/1033.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250602%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250602T173233Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=96d108cf9697780cecec87d16ef377f7d579e073c21c46dd1e38142f74402d046d7c39d0a4c5913906cebb4e853d2180e622f442b1ce79cbf1ac867759a9c1a01d9923d3be9b3c1e709940698c968502d6f3ef6879cfbaabd1ae074128c289477fe8e769296a0f818e69071b159bbf567a3f3f369398e39b7b6a805f1cd09eab63b269fcfbb8bf939f7da3e554c057214dfd55a79fe8ebe0afd1dd5bf3cf53836c8676875c1a89056f74337fa8a9240fc2dd0219535f7d4f7bc597397dfdfd7d29a0f8a3c67a66f74013b8b11dabf4e541abb190cabd4f22c714de5b2c5cf343439547e25d28bb27cdd528052bbcb86b74ccaef0e247c7ca3c5a9aeba1f83273");
		img.getWrappedImage();
		
		Path modelDir = Paths.get("build/mlp");
		Model model = Model.newInstance("mlp");
		model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
		model.load(modelDir);
		
		Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

		    @Override
		    public NDList processInput(TranslatorContext ctx, Image input) {
		        // Convert Image to NDArray
		        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
		        return new NDList(NDImageUtils.toTensor(array));
		    }

		    @Override
		    public Classifications processOutput(TranslatorContext ctx, NDList list) {
		        // Create a Classifications with the output probabilities
		        NDArray probabilities = list.singletonOrThrow().softmax(0);
		        List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
		        return new Classifications(classNames, probabilities);
		    }

		    @Override
		    public Batchifier getBatchifier() {
		        return Batchifier.STACK;
		    }
		};

		Predictor<Image, Classifications> predictor = model.newPredictor(translator);
		Classifications classifications = predictor.predict(img);

		System.out.println(classifications);

	}

}
