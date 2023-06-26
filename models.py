
def train_svm_model(features, labels):
    labels = labels.astype(np.float32)

    # Reshape features to have a consistent shape
    features = features.reshape(features.shape[0], -1)

    # Create an SVM classifier
    svm = SVC()

    # Create a TensorFlow graph for logging summaries
    graph = tf.Graph()
    with graph.as_default():
        # Define placeholders for input and labels
        input_placeholder = tf.placeholder(tf.float32, shape=[None, features.shape[1]], name='input')
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, labels.shape[1]], name='labels')

        # Fit the SVM classifier to the training data
        svm.fit(features, labels)

        # Create a summary writer to write the training summaries
        log_dir = './logs'  # Specify the directory for saving the log files
        summary_writer = tf.summary.FileWriter(log_dir, graph)

        # Training loop
        print('Training model...')
        num_steps = 1000
        for i in range(num_steps):
            features_batch, labels_batch = _get_examples_batch(
                features, labels, FLAGS.batch_size)

            # Fit the SVM classifier to the training data
            svm.fit(features_batch, labels_batch)

            if i % 100 == 0:
                # Evaluate the SVM classifier on the training data
                accuracy = svm.score(features_batch, labels_batch)
                print('Step %d: accuracy %g' % (i, accuracy))

                # Create summary objects for accuracy
                accuracy_summary = tf.Summary()
                accuracy_summary.value.add(tag='Accuracy', simple_value=accuracy)

                # Write the accuracy summary to the log file
                summary_writer.add_summary(accuracy_summary, i)

        # Close the summary writer
        summary_writer.close()

def train_vggish_model(features, labels):
    labels = labels.astype(np.float32)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish.
        embeddings = vggish_slim.define_vggish_slim(training=FLAGS.train_vggish)

        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        with tf.variable_scope('mymodel'):
            # Add a fully connected layer with 100 units. Add an activation function
            # to the embeddings since they are pre-activation.
            num_units = 100
            fc = slim.fully_connected(
                embeddings,
                num_units,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.initializers.variance_scaling)
            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
                fc,
                _NUM_CLASSES,
                activation_fn=None,
                weights_initializer=tf.initializers.variance_scaling)

            # Add training ops.
            global_step = tf.train.create_global_step()
            with tf.variable_scope('train'):
                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                # Loss is calculated as the sigmoid cross entropy between the
                # predictions and the true labels.
                xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels)
                loss = tf.reduce_mean(xent)
                train_op = tf.train.AdamOptimizer().minimize(
                    loss, global_step=global_step, name='train_op')

        # Initialize all variables in the model.
        sess.run(tf.global_variables_initializer())

        # Locate all the tensors and ops we need for the training loop.
        with tf.variable_scope('mymodel/train'):
            global_step = tf.train.create_global_step()

        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        labels_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/Mean:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

        # Training loop.
        print('Training model...')
        num_steps = 1000
        for i in range(num_steps):
            [features_batch, labels_batch] = _get_examples_batch(
                features, labels, FLAGS.batch_size)
            if i % 100 == 0:
                [num_steps, loss_value, _] = sess.run(
                    [global_step_tensor, loss_tensor, train_op],
                    feed_dict={features_tensor: features_batch,
                               labels_tensor: labels_batch})
                print('Step %d: loss %g' % (num_steps, loss_value))
            else:
                sess.run(train_op,
                         feed_dict={features_tensor: features_batch,
                                    labels_tensor: labels_batch})
