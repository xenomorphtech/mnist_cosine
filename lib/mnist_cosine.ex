defmodule MNISTCosine do
  import Nx.Defn

  @zero Nx.tensor(0.0)

  @batch_size 20
  @batches 25
  @total_input (@batch_size * @batches)

  @vector_dims 25

  @batch_shape_img [@batch_size, 1, 28, 28]
  @batch_shape_labels [@batch_size, @vector_dims]

  @vector_count 10
  @similarity 0.5

  @epochs 20

  defn check_rng() do Nx.random_normal({3}) end

  def axon do
    model = 
    Axon.input({nil, 1, 28, 28})
    |> Axon.conv(16, kernel_size: {3, 3}, padding: [{1,1},{1,1}], activation: :relu)
    |> Axon.max_pool(kernel_size: 2)
    |> Axon.conv(8, kernel_size: {3, 3}, padding: [{2,2},{2,2}], activation: :relu)
    |> Axon.max_pool(kernel_size: 2)
    |> Axon.conv(1, kernel_size: {3, 3}, padding: [{2,2},{2,2}], activation: :relu)
    |> Axon.max_pool(kernel_size: 2)
  end

  def init_weights() do
    w1 = Axon.Initializers.glorot_uniform(shape: {16, 1, 3, 3})
    b1 = Axon.Initializers.zeros(shape: {16})
    w2 = Axon.Initializers.glorot_uniform(shape: {8, 16, 3, 3})
    b2 = Axon.Initializers.zeros(shape: {8})
    w3 = Axon.Initializers.glorot_uniform(shape: {1, 8, 3, 3})
    b3 = Axon.Initializers.zeros(shape: {1})
    binding() |> Enum.into(%{})
  end

  defn predict(w, input) do
    input
    |> Axon.Layers.conv(w.w1, w.b1, padding: [{1,1},{1,1}])
    |> Axon.Activations.relu()
    |> Axon.Layers.max_pool(kernel_size: 2)
    |> Axon.Layers.conv(w.w2, w.b2, padding: [{2,2},{2,2}])
    |> Axon.Activations.relu()
    |> Axon.Layers.max_pool(kernel_size: 2)
    |> Axon.Layers.conv(w.w3, w.b3, padding: [{2,2},{2,2}])
    |> Axon.Activations.relu()
    |> Axon.Layers.max_pool(kernel_size: 2)
  end

  def label_vectors() do
      Nx.tensor [
        [0.0788, 0.1254, 0.0268, 0.2183, 0.3003, 0.3279, 0.0510, 0.0439, 0.3552,
          0.0749, 0.0862, 0.3351, 0.0506, 0.1168, 0.1643, 0.1960, 0.2681, 0.2267,
          0.1867, 0.3841, 0.0528, 0.1355, 0.0233, 0.0842, 0.2447], # digit 0
        [0.0751, 0.1258, 0.2656, 0.0132, 0.0048, 0.0459, 0.0123, 0.0921, 0.0722,
          0.4720, 0.0563, 0.0858, 0.1946, 0.0289, 0.0549, 0.0889, 0.0180, 0.0328,
          0.2810, 0.0050, 0.2128, 0.0508, 0.6535, 0.0418, 0.2278],  # digit 1
        [0.2016, 0.2286, 0.0246, 0.0350, 0.0284, 0.0113, 0.5795, 0.0695, 0.0989,
          0.0515, 0.5213, 0.0253, 0.2721, 0.0178, 0.0095, 0.0061, 0.0243, 0.0309,
          0.1076, 0.0285, 0.2499, 0.1712, 0.0729, 0.3033, 0.0366],  # digit 2
        [0.0282, 0.1384, 0.0564, 0.0931, 0.0529, 0.0337, 0.1164, 0.1348, 0.0838,
          0.0159, 0.3105, 0.0114, 0.1378, 0.0378, 0.0598, 0.7858, 0.0471, 0.1663,
          0.0565, 0.0347, 0.0680, 0.2686, 0.0610, 0.2588, 0.0746],  # digit 3
        [0.0368, 0.1348, 0.6532, 0.1854, 0.0323, 0.1109, 0.1187, 0.0222, 0.0124,
          0.1473, 0.2992, 0.0402, 0.0864, 0.2666, 0.0860, 0.0477, 0.2516, 0.2643,
          0.0793, 0.1082, 0.3297, 0.0828, 0.0344, 0.0016, 0.1499],  # digit 4
        [0.4528, 0.1523, 0.0251, 0.0511, 0.1991, 0.0561, 0.1557, 0.6983, 0.0731,
          0.1860, 0.0884, 0.0276, 0.1971, 0.1252, 0.1858, 0.0104, 0.0453, 0.1107,
          0.0252, 0.0585, 0.0470, 0.1628, 0.1336, 0.0263, 0.1031],  # digit 5
        [0.1442, 0.0414, 0.0260, 0.1051, 0.1011, 0.0484, 0.0940, 0.0715, 0.0459,
          0.5043, 0.0312, 0.0336, 0.1456, 0.5837, 0.2772, 0.0289, 0.4086, 0.0950,
          0.1037, 0.0274, 0.0142, 0.0024, 0.0353, 0.2247, 0.0419],  # digit 6
        [0.0253, 0.0100, 0.0521, 0.1290, 0.1214, 0.1809, 0.0092, 0.0036, 0.0223,
          0.0029, 0.0192, 0.0209, 0.5189, 0.1902, 0.2108, 0.0400, 0.0302, 0.1245,
          0.4193, 0.0728, 0.0449, 0.5350, 0.2411, 0.1886, 0.0111],  # digit 7
        [0.1489, 0.7321, 0.0011, 0.0065, 0.0215, 0.0462, 0.1617, 0.0958, 0.1073,
          0.0546, 0.0956, 0.0459, 0.0222, 0.1168, 0.2183, 0.0008, 0.0317, 0.0330,
          0.0749, 0.3990, 0.0605, 0.1267, 0.3394, 0.1195, 0.0142],  # digit 8
        [0.5247, 0.0227, 0.2495, 0.0037, 0.1025, 0.2442, 0.0776, 0.0051, 0.0561,
          0.0639, 0.0168, 0.1553, 0.1897, 0.0241, 0.0676, 0.1564, 0.0050, 0.0364,
          0.0179, 0.0769, 0.0420, 0.0796, 0.2302, 0.5864, 0.2758]  # digit 9
      ]
  end

  def label_vectors_cached() do
    lv = :persistent_term.get(:label_vectors, nil)
    if lv do lv else
        lv = generate_label_vectors_n()
        :persistent_term.put(:label_vectors, lv)
        lv
    end
    #label_vectors()
  end

  defn gen_vec() do
    vec = Nx.random_normal({@vector_dims})
    Nx.power(Nx.power(vec, 2) / Nx.sum(Nx.power(vec, 2)), 0.5)
  end

  build_recursive = fn(build_recursive, idx)->
    nest = if idx == (@vector_count-1) do 0 else build_recursive.(build_recursive, idx+1) end
    """
    if found >= #{idx} do
      sim = MNISTCosine.loss_cosine_similarity_torch(Nx.new_axis(new_vec,0), Nx.new_axis(labels[#{idx}],0))
      if sim[0] > #{@similarity} do
        1
      else
        #{nest}
      end
    else 0 end
    """
  end
  Code.compile_string("""
    defmodule Unroller do
      import Nx.Defn
      defn check_sim(found, labels, new_vec) do
        #{build_recursive.(build_recursive, 0)}
      end
    end
  """)

  defn add_label(labels, row, new_vec) do
    indices = Nx.stack([
      Nx.broadcast(row, {@vector_dims}),
      Nx.iota({@vector_dims})
    ], axis: -1)
    labels = Nx.indexed_add(labels, indices, new_vec)
  end

  defn generate_label_vectors_n() do
    labels = Nx.broadcast(0, {@vector_count, @vector_dims})
    labels = add_label(labels, 0, gen_vec())
    found = 0
    {_, labels} = while {found, labels}, Nx.less(found, @vector_count) do
      new_vec = gen_vec()
      is_similar = Unroller.check_sim(found, labels, new_vec)
      if not is_similar do
        labels = add_label(labels, found, new_vec)
        {found + 1, labels}
      else
        {found, labels}
      end
    end
    labels
  end

  defn generate_label_vectors_n_2() do
    labels = Nx.broadcast(0, {@vector_count, @vector_dims})
    labels = add_label(labels, 0, gen_vec())
    found = 0
    {_, labels} = while {found, labels}, Nx.less(found, @vector_count) do
      new_vec = gen_vec()
      is_similar = Unroller.check_sim(found, labels, new_vec)
      if not is_similar do
        labels = add_label(labels, found, new_vec)
        {found + 1, labels}
      else
        {found, labels}
      end
    end
  end

  def generate_label_vectors() do
    labels = [gen_vec()]
    Enum.reduce_while(1..300_000, labels, fn(_, labels)->
      v = gen_vec()
      is_similar = Enum.find(labels, fn(l)->
        sim = loss_cosine_similarity_torch(Nx.new_axis(v,0), Nx.new_axis(l,0))
        Nx.to_number(sim[0]) > @similarity
      end)
      cond do
        length(labels) >= @vector_count -> {:halt, labels}
        !is_similar ->
          IO.inspect "found"
          {:cont, labels ++ [v]}
        true -> {:cont, labels}
      end
    end)
  end

  defn loss(w, batch_images, batch_labels) do
    preds = predict(w, batch_images)
    |> Nx.reshape({@batch_size, @vector_dims})
    loss_cosine_similarity(batch_labels, preds)
  end

  defn loss_cosine_similarity_torch(x1, x2) do
    dim = [1]
    eps = 0.000001

    w12 = Nx.sum(x1 * x2, axes: [1])
    w1 = Nx.sum(x1 * x1, axes: [1])
    w2 = Nx.sum(x2 * x2, axes: [1])
    n12 = Nx.sqrt(Nx.max((w1 * w2), eps*eps))

    #w12 = Nx.sum(x1 * x2, axes: dim)
    #w1 = Nx.LinAlg.norm(x1, axes: dim)
    #w2 = Nx.LinAlg.norm(x2, axes: dim)
    #n12 = Nx.max(w1 * w2, eps)

    w12 / n12
  end

  defn loss_cosine_similarity(x1, x2) do
    Nx.sum(1 - loss_cosine_similarity_torch(x1, x2)) / Nx.axis_size(x1, 0)
  end

  defn objective(w, batch_images, batch_labels) do
    preds = predict(w, batch_images)
    |> Nx.reshape({@batch_size, 25})
    loss = loss_cosine_similarity(preds, batch_labels)
    {preds, loss}
  end

  defn update_with_averages(m, imgs, labels, update_fn) do
    w = m.w
    {{preds, loss}, gw} = value_and_grad(w, &objective(&1, imgs, labels), &elem(&1, 1))
    {scaled_updates, optimizer_state} = update_fn.(gw, m.optimizer_state, w)
    w = Axon.Updates.apply_updates(w, scaled_updates)

    avg_loss = m.loss + (loss * @batch_size) / @total_input

    %{m | w: w, optimizer_state: optimizer_state, loss: avg_loss}
  end

  defn train_epoch_2(m, imgs, labels, update_fn) do
    batches = @batches - 1
    {_, m, _, _} = while {batches, m, imgs, labels}, Nx.greater_equal(batches,0) do
        img_slice = Nx.slice(imgs, [@batch_size*batches,0,0,0], @batch_shape_img)
        label_slice = Nx.slice(labels, [@batch_size*batches,0], @batch_shape_labels)
        m = update_with_averages(m, img_slice, label_slice, update_fn)
        {batches - 1, m, imgs, labels}
    end
    m
  end

  def train_digit(imgs, labels) do
    w = MNISTCosine.init_weights()
    {init_fn, update_fn} = Axon.Optimizers.adamw(0.01, decay: 0.01)
    optimizer_state = init_fn.(w)
    m = %{optimizer_state: optimizer_state, w: w, loss: @zero}

    Enum.reduce(1..@epochs, m, fn(_, m) ->
        m = %{m | loss: @zero}
        train_epoch_2(m, imgs, labels, update_fn)
    end)
  end

  def go() do
    Nx.Defn.global_default_options(compiler: EXLA, client: :host)

    set = MNISTCosine.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

    by_digit = set
    |> Enum.group_by(& &1.digit)
    |> Enum.into(%{}, fn{digit, set}->
        {digit, Enum.take(Enum.shuffle(set), @total_input)}
    end)

    Enum.into(0..9, %{}, fn(n)->
        digit = by_digit[n]
        imgs = digit
        |> Enum.reduce("", & &2 <> &1.img)
        |> Nx.from_binary({:u, 8})
        |> Nx.reshape({@total_input, 1, 28, 28})
        |> Nx.divide(255)

        magic_labels = MNISTCosine.label_vectors_cached()
        label = magic_labels[n]
        labels = Enum.map(1..@total_input, fn(_)-> Nx.to_flat_list(label) end)
        |> Nx.tensor()

        m = train_digit(imgs, labels)
        {n, m}
    end)
  end

  def download(images, labels) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      unzip_cache_or_download(images)
    <<_::32, n_labels::32, labels::binary>> = unzip_cache_or_download(labels)
    images = Enum.map(0..(n_images-1), fn(idx)->
        :binary.part(images, idx*n_cols*n_rows, n_cols*n_rows)
    end)
    labels = Enum.map(0..(n_labels-1), fn(idx)->
        :binary.part(labels, idx, 1)
    end)
    set = Enum.zip(images, labels)
    |> Enum.map(fn{img,<<n>>}-> %{img: img, digit: n} end)
  end

  def unzip_cache_or_download(zip) do
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    path = Path.join("tmp", zip)

    data =
      if File.exists?(path) do
        IO.puts("Using #{zip} from tmp/\n")
        File.read!(path)
      else
        IO.puts("Fetching #{zip} from https://storage.googleapis.com/cvdf-datasets/mnist/\n")
        :inets.start()
        :ssl.start()

        {:ok, {_status, _response, data}} = :httpc.request(base_url ++ zip)
        File.mkdir_p!("tmp")
        File.write!(path, data)

        data
      end

    :zlib.gunzip(data)
  end

  def test(m) do
    Nx.Defn.global_default_options(compiler: EXLA, client: :host)

    train_images = MNISTCosine.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    test_images = MNISTCosine.download('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    train_correct = Enum.chunk_every(train_images, 1000)
    |> Enum.map(& test_1(&1, m))
    |> Enum.sum()
    train_perc = Float.round((train_correct / 60000)*100, 6)
    test_correct = Enum.chunk_every(test_images, 1000)
    |> Enum.map(& test_1(&1, m))
    |> Enum.sum()
    test_perc = Float.round((test_correct / 10000)*100, 6)
    "Train Set: #{train_correct}/60000 (#{train_perc}%) | " <>
    "Test Set: #{test_correct}/10000 (#{test_perc}%)"
  end

  def test_1(images, m) do
    magic_labels = MNISTCosine.label_vectors_cached()
    imgs = images
    |> Enum.reduce("", & &2 <> &1.img)
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({1000, 1, 28, 28})
    |> Nx.divide(255)

    labels = Enum.map(images, fn(%{digit: d})->
        Nx.to_flat_list(magic_labels[d]) 
    end)
    |> Nx.tensor()

    loss_list = Enum.map(m, fn({digit, %{w: w}})->
        preds = MNISTCosine.predict(w, imgs)
        |> Nx.reshape({1000, @vector_dims})
        MNISTCosine.loss_cosine_similarity_torch(preds, labels)
    end)

    loss = Nx.stack(loss_list)
    |> Nx.transpose()
    |> Nx.argsort(axis: 1)

    preds = Nx.slice(loss, [0,10], [1000,1])
    |> Nx.to_flat_list()
    correct = Enum.reduce(Enum.zip(images, preds), 0, fn({%{digit: d}, pred}, acc)->
        if d == pred do acc+1 else acc end
    end)
  end
end