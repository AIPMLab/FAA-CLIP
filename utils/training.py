import torch
from adaptation import LMMDLoss
from utils.clip_util import AverageMeter
import utils.clip_util as clu
import torch.nn as nn
import clip
from utils.loss_function import CrossEntropyLabelSmooth
from utils.clip_util import convert_models_to_fp32
from utils.clip_util import FocalLossWithSmoothing
from tqdm import tqdm

def totrain(model):
    model.model.train()
    model.fea_attn.train()


def train(args, model, data_loader, optimizer, device, testloader, mmd_loss, server_model, previous_nets):
    totrain(model)
    texts = model.labels
    t_features = clu.get_text_features_list(texts, model.model).float() #update each round based on the model
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    train_loss_clf = AverageMeter()
    train_loss_transfer = AverageMeter()
    print(len(data_loader), len(testloader))
    source_data = iter(data_loader)
    target_data = iter(testloader)
    loss_all = 0
    if args.method == 'ours':
        if args.dataset == 'BrainTumor':
            for _ in (range(0, args.n_iter)):
                image, text, label = next(source_data)  # .next()
                image_t, _, _ = next(target_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    image_t = image_t.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()
                    image_features_att = model.fea_attn(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    test_features = model.model.encode_image(image_t).float()
                    with torch.no_grad():
                        test_features_att = model.fea_attn(test_features)
                        test_features = torch.mul(test_features_att, test_features)

                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)
                    test_features = test_features / \
                                    test_features.norm(dim=1, keepdim=True)

                    loss_m = mmd_loss(image_features, test_features)
                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    cla_loss = (loss_img(logits_per_image, ground_truth) +
                                loss_txt(logits_per_text, ground_truth)) / 2
                    loss = cla_loss + 0.5 * loss_m

                    train_loss_clf.update(cla_loss.item())
                    train_loss_transfer.update(0.5 * loss_m.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)
        elif args.dataset == 'RealSkin' or args.dataset == 'Dermnet' or args.dataset == 'havior' or args.dataset == 'OfficeHome' or args.dataset == 'ModernOffice31':
            for _ in (range(0, len(data_loader))):
                image, text, label = next(source_data)  # .next()
                image_t, _, _ = next(target_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    image_t = image_t.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()
                    image_features_att = model.fea_attn(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    test_features = model.model.encode_image(image_t).float()
                    with torch.no_grad():
                        test_features_att = model.fea_attn(test_features)
                        test_features = torch.mul(test_features_att, test_features)

                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)
                    test_features = test_features / \
                                    test_features.norm(dim=1, keepdim=True)

                    loss_m = mmd_loss(image_features, test_features)
                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    cla_loss = (loss_img(logits_per_image, ground_truth) +
                                loss_txt(logits_per_text, ground_truth)) / 2
                    loss = cla_loss + 0.5 * loss_m

                    train_loss_clf.update(cla_loss.item())
                    train_loss_transfer.update(0.5 * loss_m.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)

    if args.method == 'fedprox':
        if args.dataset == 'BrainTumor':
            for _ in tqdm(range(0, args.n_iter)):
                image, text, label = next(source_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()
                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)
                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    loss = (loss_img(logits_per_image, ground_truth) +
                            loss_txt(logits_per_text, ground_truth)) / 2
                    train_loss_clf.update(loss.item())
                    # print(loss)
                    # loss_all += loss
                    if args.step > 0:
                        w_diff = torch.tensor(1e-10, device=device)
                        for w, w_t in zip(server_model.parameters(), model.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2).float()  # model difference
                            # print(w_diff)
                        w_diff = torch.sqrt(w_diff)
                        train_loss_transfer.update((1e-2 / 2. * w_diff).item())
                        loss += 1e-2 / 2. * w_diff  # dif loss
                        # print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)
            print("cla loss: ", train_loss_clf.avg, 'w_diff loss: ', train_loss_transfer.avg)
        elif args.dataset == 'RealSkin' or args.dataset == 'Dermnet' or args.dataset == 'havior' or args.dataset == 'OfficeHome' or args.dataset == 'ModernOffice31':
            for _ in tqdm(range(0, len(data_loader))):
                image, text, label = next(source_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()
                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)
                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    loss = (loss_img(logits_per_image, ground_truth) +
                            loss_txt(logits_per_text, ground_truth)) / 2
                    train_loss_clf.update(loss.item())
                    # print(loss)
                    # loss_all += loss
                    if args.step > 0:
                        w_diff = torch.tensor(1e-10, device=device)
                        for w, w_t in zip(server_model.parameters(), model.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2).float()  # model difference
                            # print(w_diff)
                        w_diff = torch.sqrt(w_diff)
                        train_loss_transfer.update((1e-2 / 2. * w_diff).item())
                        loss += 1e-2 / 2. * w_diff  # dif loss
                        # print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)
            print("cla loss: ", train_loss_clf.avg, 'w_diff loss: ', train_loss_transfer.avg)
    if args.method == 'moon':
        cnt = 0
        cos = torch.nn.CosineSimilarity(dim=-1)
        criterion = nn.CrossEntropyLoss()
        mu = 1
        if args.dataset == 'BrainTumor':
            for _ in tqdm(range(0, args.n_iter)):
                image, text, label = next(source_data)
                optimizer.zero_grad()
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                image_features_glo = server_model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                image_features_glo = image_features_glo / \
                                     image_features_glo.norm(dim=1, keepdim=True)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                # logits_per_image_glo = logit_scale * image_features_glo @ text_features_glo.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
                train_loss_clf.update(loss.item())
                # MOON contrastive loss below, we refered the original codes, it needs [logits_per_image] to measure.
                # Model-Contrastive Federated Learning
                posi = cos(image_features, image_features_glo)
                logits = posi.reshape(-1, 1)
                if args.step > 0:
                    image_features_pre = previous_nets.model.encode_image(image).float()
                    # text_features_pre = previous_nets.model.encode_text(text).float()
                    image_features_pre = image_features_pre / \
                                         image_features_pre.norm(dim=1, keepdim=True)
                    nega = cos(image_features, image_features_pre)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                    logits /= args.temp
                    labels = torch.zeros(image.size(0)).cuda().long()
                    loss += mu * criterion(logits, labels)
                    train_loss_transfer.update(mu * criterion(logits, labels))
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            print("cla loss: ", train_loss_clf.avg, 'MOON loss: ', train_loss_transfer.avg)
        elif args.dataset == 'RealSkin' or args.dataset == 'Dermnet' or args.dataset == 'havior' or args.dataset == 'OfficeHome' or args.dataset == 'ModernOffice31':
            for _ in tqdm(range(0, len(data_loader))):
                image, text, label = next(source_data)
                optimizer.zero_grad()
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                image_features_glo = server_model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                image_features_glo = image_features_glo / \
                                     image_features_glo.norm(dim=1, keepdim=True)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                # logits_per_image_glo = logit_scale * image_features_glo @ text_features_glo.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
                train_loss_clf.update(loss.item())
                # MOON contrastive loss below, we refered the original codes, it needs [logits_per_image] to measure.
                # Model-Contrastive Federated Learning
                posi = cos(image_features, image_features_glo)
                logits = posi.reshape(-1, 1)
                if args.step > 0:
                    image_features_pre = previous_nets.model.encode_image(image).float()
                    # text_features_pre = previous_nets.model.encode_text(text).float()
                    image_features_pre = image_features_pre / \
                                         image_features_pre.norm(dim=1, keepdim=True)
                    nega = cos(image_features, image_features_pre)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                    logits /= args.temp
                    labels = torch.zeros(image.size(0)).cuda().long()
                    loss += mu * criterion(logits, labels)
                    train_loss_transfer.update(mu * criterion(logits, labels))
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            print("cla loss: ", train_loss_clf.avg, 'MOON loss: ', train_loss_transfer.avg)

    if args.method == 'fedclip':
        if args.dataset == 'BrainTumor':
            for _ in tqdm(range(0, args.n_iter)):
                image, text, label = next(source_data)  # .next()
                # image_t, _, _ = next(target_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()
                    image_features_att = model.fea_attn(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)

                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    cla_loss = (loss_img(logits_per_image, ground_truth) +
                                loss_txt(logits_per_text, ground_truth)) / 2

                    train_loss_clf.update(cla_loss.item())
                    optimizer.zero_grad()
                    cla_loss.backward()
                    optimizer.step()
            print("cla loss: ", train_loss_clf.avg)
        elif args.dataset == 'RealSkin' or args.dataset == 'Dermnet' or args.dataset == 'havior' or args.dataset == 'OfficeHome' or args.dataset == 'ModernOffice31':
            for _ in (range(0, len(data_loader))):
                image, text, label = next(source_data)  # .next()
                # image_t, _, _ = next(target_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()
                    image_features_att = model.fea_attn(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)

                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    cla_loss = (loss_img(logits_per_image, ground_truth) +
                                loss_txt(logits_per_text, ground_truth)) / 2

                    train_loss_clf.update(cla_loss.item())
                    optimizer.zero_grad()
                    cla_loss.backward()
                    optimizer.step()
            print("cla loss: ", train_loss_clf.avg)

    if args.method == 'fedavg':
        if args.dataset == 'BrainTumor':
            for _ in tqdm(range(0, args.n_iter)):
                image, text, label = next(source_data)  # .next()
                # image_t, _, _ = next(target_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    # image_t = image_t.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()

                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)

                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    cla_loss = (loss_img(logits_per_image, ground_truth) +
                                loss_txt(logits_per_text, ground_truth)) / 2

                    train_loss_clf.update(cla_loss.item())
                    optimizer.zero_grad()
                    cla_loss.backward()
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)
            print("cla loss: ", train_loss_clf.avg)
        elif args.dataset == 'RealSkin' or args.dataset == 'Dermnet' or args.dataset == 'havior' or args.dataset == 'OfficeHome' or args.dataset == 'ModernOffice31':
            for _ in tqdm(range(0, len(data_loader))):
                image, text, label = next(source_data)  # .next()
                # image_t, _, _ = next(target_data)  # .next()
                if len(text) > 1:
                    image = image.to(device)
                    text = text.to(device)
                    # image_t = image_t.to(device)
                    image_features = model.model.encode_image(image).float()
                    text_features = model.model.encode_text(text).float()

                    image_features = image_features / \
                                     image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / \
                                    text_features.norm(dim=1, keepdim=True)

                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(
                        len(image), dtype=torch.long, device=device)

                    cla_loss = (loss_img(logits_per_image, ground_truth) +
                                loss_txt(logits_per_text, ground_truth)) / 2

                    train_loss_clf.update(cla_loss.item())
                    optimizer.zero_grad()
                    cla_loss.backward()
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)
            print("cla loss: ", train_loss_clf.avg)
