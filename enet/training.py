import os
import torch
from math import ceil

from torchtext.data import BucketIterator

from enet.util import run_over_data


def train(model, train_set, dev_set, test_set, optimizer_constructor, epochs, tester, parser, other_testsets):
    # build batch on cpu
    pt_version = torch.__version__.rsplit('.', 1)
    if float(pt_version[0]) < 1.1:
        dev = -1
    else:
        dev = model.device
    train_iter = BucketIterator(train_set, batch_size=parser.batch, train=False, shuffle=True, device=dev,
                                sort_key=lambda x: len(x.POSTAGS))
    dev_iter = BucketIterator(dev_set, batch_size=parser.batch, train=False, shuffle=True, device=dev,
                              sort_key=lambda x: len(x.POSTAGS))
    test_iter = BucketIterator(test_set, batch_size=parser.batch, train=False, shuffle=True, device=dev,
                               sort_key=lambda x: len(x.POSTAGS))

    scores = 0.0
    now_bad = 0
    restart_used = 0
    print("\nStarting training...\n")
    lr = parser.lr
    optimizer = optimizer_constructor(lr=lr)

    for i in range(epochs):
        # Training Phrase
        print("Epoch", i + 1)
        training_loss, training_ed_p, training_ed_r, training_ed_f1, \
        training_ae_p, training_ae_r, training_ae_f1 = run_over_data(data_iter=train_iter,
                                                                     optimizer=optimizer,
                                                                     model=model,
                                                                     need_backward=True,
                                                                     MAX_STEP=ceil(len(train_set) / parser.batch),
                                                                     tester=tester,
                                                                     hyps=model.hyperparams,
                                                                     device=model.device,
                                                                     maxnorm=parser.maxnorm,
                                                                     word_i2s=parser.word_i2s,
                                                                     label_i2s=parser.label_i2s,
                                                                     role_i2s=parser.role_i2s,
                                                                     weight=parser.label_weight,
                                                                     save_output=os.path.join(parser.out,
                                                                                              "training_epoch_%d.txt" % (
                                                                                                  i + 1)))
        print("\nEpoch", i + 1, " training loss: ", training_loss,
              "\ntraining ed p: ", training_ed_p,
              " training ed r: ", training_ed_r,
              " training ed f1: ", training_ed_f1,
              "\ntraining ae p: ", training_ae_p,
              " training ae r: ", training_ae_r,
              " training ae f1: ", training_ae_f1)
        parser.writer.add_scalar('train/loss', training_loss, i)
        parser.writer.add_scalar('train/ed/p', training_ed_p, i)
        parser.writer.add_scalar('train/ed/r', training_ed_r, i)
        parser.writer.add_scalar('train/ed/f1', training_ed_f1, i)
        parser.writer.add_scalar('train/ae/p', training_ae_p, i)
        parser.writer.add_scalar('train/ae/r', training_ae_r, i)
        parser.writer.add_scalar('train/ae/f1', training_ae_f1, i)

        # Validation Phrase
        dev_loss, dev_ed_p, dev_ed_r, dev_ed_f1, \
        dev_ae_p, dev_ae_r, dev_ae_f1 = run_over_data(data_iter=dev_iter,
                                                      optimizer=optimizer,
                                                      model=model,
                                                      need_backward=False,
                                                      MAX_STEP=ceil(len(dev_set) / parser.batch),
                                                      tester=tester,
                                                      hyps=model.hyperparams,
                                                      device=model.device,
                                                      maxnorm=parser.maxnorm,
                                                      word_i2s=parser.word_i2s,
                                                      label_i2s=parser.label_i2s,
                                                      role_i2s=parser.role_i2s,
                                                      weight=parser.label_weight,
                                                      save_output=os.path.join(parser.out,
                                                                               "dev_epoch_%d.txt" % (
                                                                                   i + 1)))
        print("\nEpoch", i + 1, " dev loss: ", dev_loss,
              "\ndev ed p: ", dev_ed_p,
              " dev ed r: ", dev_ed_r,
              " dev ed f1: ", dev_ed_f1,
              "\ndev ae p: ", dev_ae_p,
              " dev ae r: ", dev_ae_r,
              " dev ae f1: ", dev_ae_f1)
        parser.writer.add_scalar('dev/loss', dev_loss, i)
        parser.writer.add_scalar('dev/ed/p', dev_ed_p, i)
        parser.writer.add_scalar('dev/ed/r', dev_ed_r, i)
        parser.writer.add_scalar('dev/ed/f1', dev_ed_f1, i)
        parser.writer.add_scalar('dev/ae/p', dev_ae_p, i)
        parser.writer.add_scalar('dev/ae/r', dev_ae_r, i)
        parser.writer.add_scalar('dev/ae/f1', dev_ae_f1, i)

        # Testing Phrase
        test_loss, test_ed_p, test_ed_r, test_ed_f1, \
        test_ae_p, test_ae_r, test_ae_f1 = run_over_data(data_iter=test_iter,
                                                         optimizer=optimizer,
                                                         model=model,
                                                         need_backward=False,
                                                         MAX_STEP=ceil(len(test_set) / parser.batch),
                                                         tester=tester,
                                                         hyps=model.hyperparams,
                                                         device=model.device,
                                                         maxnorm=parser.maxnorm,
                                                         word_i2s=parser.word_i2s,
                                                         label_i2s=parser.label_i2s,
                                                         role_i2s=parser.role_i2s,
                                                         weight=parser.label_weight,
                                                         save_output=os.path.join(parser.out,
                                                                                  "test_epoch_%d.txt" % (
                                                                                      i + 1)))
        print("\nEpoch", i + 1, " test loss: ", test_loss,
              "\ntest ed p: ", test_ed_p,
              " test ed r: ", test_ed_r,
              " test ed f1: ", test_ed_f1,
              "\ntest ae p: ", test_ae_p,
              " test ae r: ", test_ae_r,
              " test ae f1: ", test_ae_f1)
        parser.writer.add_scalar('test/loss', test_loss, i)
        parser.writer.add_scalar('test/ed/p', test_ed_p, i)
        parser.writer.add_scalar('test/ed/r', test_ed_r, i)
        parser.writer.add_scalar('test/ed/f1', test_ed_f1, i)
        parser.writer.add_scalar('test/ae/p', test_ae_p, i)
        parser.writer.add_scalar('test/ae/r', test_ae_r, i)
        parser.writer.add_scalar('test/ae/f1', test_ae_f1, i)

        # Early Stop
        if scores <= dev_ed_f1 + dev_ae_f1:
            scores = dev_ed_f1 + dev_ae_f1
            # Move model parameters to CPU
            model.save_model(os.path.join(parser.out, "model.pt"))
            print("Save CPU model at Epoch", i + 1)
            now_bad = 0
        else:
            now_bad += 1
            if now_bad >= parser.earlystop:
                if restart_used >= parser.restart:
                    print("Restart opportunity are run out")
                    break
                restart_used += 1
                print("lr decays and best model is reloaded")
                lr = lr * 0.1
                model.load_model(os.path.join(parser.out, "model.pt"))
                optimizer = optimizer_constructor(lr=lr)
                print("Restart in Epoch %d" % (i + 2))
                now_bad = 0

    # Testing Phrase
    test_loss, test_ed_p, test_ed_r, test_ed_f1, \
    test_ae_p, test_ae_r, test_ae_f1 = run_over_data(data_iter=test_iter,
                                                     optimizer=optimizer,
                                                     model=model,
                                                     need_backward=False,
                                                     MAX_STEP=ceil(len(test_set) / parser.batch),
                                                     tester=tester,
                                                     hyps=model.hyperparams,
                                                     device=model.device,
                                                     maxnorm=parser.maxnorm,
                                                     word_i2s=parser.word_i2s,
                                                     label_i2s=parser.label_i2s,
                                                     role_i2s=parser.role_i2s,
                                                     weight=parser.label_weight,
                                                     save_output=os.path.join(parser.out, "test_final.txt"))
    print("\nFinally test loss: ", test_loss,
          "\ntest ed p: ", test_ed_p,
          " test ed r: ", test_ed_r,
          " test ed f1: ", test_ed_f1,
          "\ntest ae p: ", test_ae_p,
          " test ae r: ", test_ae_r,
          " test ae f1: ", test_ae_f1)

    for name, additional_test_set in other_testsets.items():
        additional_test_iter = BucketIterator(additional_test_set, batch_size=parser.batch, train=False, shuffle=True,
                                              device=-1,
                                              sort_key=lambda x: len(x.POSTAGS))

        additional_test_loss, additional_test_ed_p, additional_test_ed_r, additional_test_ed_f1, \
        additional_test_ae_p, additional_test_ae_r, additional_test_ae_f1 = run_over_data(
            data_iter=additional_test_iter,
            optimizer=optimizer,
            model=model,
            need_backward=False,
            MAX_STEP=ceil(len(additional_test_set) / parser.batch),
            tester=tester,
            hyps=model.hyperparams,
            device=model.device,
            maxnorm=parser.maxnorm,
            word_i2s=parser.word_i2s,
            label_i2s=parser.label_i2s,
            role_i2s=parser.role_i2s,
            weight=parser.label_weight,
            save_output=os.path.join(parser.out, "%s.txt") % (name))
        print("\nFor ", name, ", additional test loss: ", additional_test_loss,
              " additional ed test p: ", additional_test_ed_p,
              " additional ed test r: ", additional_test_ed_r,
              " additional ed test f1: ", additional_test_ed_f1,
              " additional ae test p: ", additional_test_ae_p,
              " additional ae test r: ", additional_test_ae_r,
              " additional ae test f1: ", additional_test_ae_f1)

    print("Training Done!")
